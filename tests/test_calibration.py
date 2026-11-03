import contextlib
import dataclasses
import io
import json
import math
import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageFilter

try:
    import numpy as np
except ImportError:  # pragma: no cover - the calib extra is optional
    raise unittest.SkipTest("Install the calib extra to test calibration")

from dnf_ocr_synth.calibration.__main__ import (
    _parse_fit,
    _parse_measured,
    line_width_fit,
    load_samples,
    main,
    phase_table,
    raid_separator_profiles,
)
from dnf_ocr_synth.calibration.align import (
    GLOW_FILL,
    PLAIN_FILL,
    best_candidate,
    estimate_text_origin,
    is_glow_row,
    raid_fill_color,
    search_alignment,
)
from dnf_ocr_synth.calibration.kernel import (
    KernelTaps,
    StandardKernel,
    align_sample,
    estimate_kernel,
    fit_gain,
    model_composite,
    native_layers,
    resample,
    tap_basis,
    tap_count,
    tent_taps,
)
from dnf_ocr_synth.calibration.layout import (
    ink_advance_layout,
    raid_layout,
)
from dnf_ocr_synth.calibration.measured import (
    run_measured_kernel,
)
from dnf_ocr_synth.calibration.metrics import (
    Reference,
    binarize_nickname_pixels,
    binarize_raid_nickname_pixels,
    binarize_raid_text_pixels,
    bt601_gray,
    compare,
    edge_profile,
    line_center,
    line_profile,
    line_width,
    otsu_threshold,
    remove_detached_raid_nickname_pixels,
    remove_raid_nickname_frame_lines,
)
from dnf_ocr_synth.calibration.pipelines import (
    DEFAULT_PIPELINE,
    ScalePipeline,
    enumerate_pipelines,
    render_with_pipeline,
)
from dnf_ocr_synth.calibration.report import (
    Fit,
    SampleInfo,
    rank_pipelines,
    residual_image,
    scale_bucket,
    write_report,
)
from dnf_ocr_synth.fonts import FontPaths
from dnf_ocr_synth.render import GlyphLayout, Renderer, _colorize, _compose

FIXTURES = Path(__file__).parent / "fixtures"
GULIM = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/gulim.ttc"
COLOR = (75, 209, 255)
PORTED_CHAINS = {
    "binarizeNicknamePixels": (binarize_nickname_pixels,),
    "binarizeRaidTextPixels": (binarize_raid_text_pixels,),
    "binarizeRaidTextPixels,removeRaidNicknameFrameLines": (
        binarize_raid_text_pixels,
        remove_raid_nickname_frame_lines,
    ),
    "binarizeRaidNicknamePixels": (binarize_raid_nickname_pixels,),
}


def synthetic_layout():
    # Two glyph-like masks: a hollow box and a solid bar, on a baseline.
    box = Image.new("L", (6, 10), 0)
    box.paste(255, (1, 0, 5, 10))
    box.paste(0, (2, 2, 4, 8))
    bar = Image.new("L", (4, 7), 0)
    bar.paste(255, (0, 0, 4, 7))

    return [box, bar], [(0, -10), (8, -7)]


def band_background(width, height):
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., 0] = 30
    rgba[..., 1] = 40
    rgba[..., 2] = np.linspace(60, 90, width, dtype=np.uint8)[None, :]
    rgba[..., 3] = 255

    return rgba


def place(background, synthetic, origin):
    canvas = Image.fromarray(background, "RGBA")
    canvas.alpha_composite(synthetic, dest=origin)

    return np.asarray(canvas)


def binary_crop(rows):
    # "#" is ink (0) and "." is background (255) in a binarized crop.
    ink = np.array([[char == "#" for char in row] for row in rows])
    rgba = np.full((*ink.shape, 4), 255, dtype=np.uint8)
    rgba[ink, :3] = 0

    return rgba


def ink_rows(rgba):
    return [
        "".join("#" if v == 0 else "." for v in row) for row in rgba[..., 0]
    ]


class PipelineTests(unittest.TestCase):
    def test_enumerates_432_distinct_pipelines(self):
        pipelines = enumerate_pipelines()
        self.assertEqual(len(pipelines), 432)
        self.assertEqual(len({p.name for p in pipelines}), 432)
        self.assertIn(DEFAULT_PIPELINE, pipelines)
        self.assertEqual(
            DEFAULT_PIPELINE.name,
            "before-o1-bilinear-premultiplied-t0-xnone-ynone",
        )

    def test_rejects_values_outside_the_space(self):
        with self.assertRaises(ValueError):
            ScalePipeline(resample="lanczos")
        with self.assertRaises(ValueError):
            ScalePipeline(texel_offset=0.25)

    def test_default_pipeline_matches_colorize_at_scale_one_and_half(self):
        masks, positions = synthetic_layout()
        composed = _compose(masks, positions)
        for scale in (1, 1.5):
            with self.subTest(scale=scale):
                expected = _colorize(composed, scale, COLOR)
                rendered = render_with_pipeline(
                    masks, positions, scale, COLOR, DEFAULT_PIPELINE
                )
                self.assertEqual(rendered.image.size, expected.size)
                self.assertEqual(rendered.image.tobytes(), expected.tobytes())
                self.assertEqual(rendered.text_origin, (2 * scale, 2 * scale))

    def test_every_pipeline_is_deterministic_and_sized_by_outline(self):
        masks, positions = synthetic_layout()
        composed = _compose(masks, positions)
        for pipeline in enumerate_pipelines():
            with self.subTest(pipeline=pipeline.name):
                first = render_with_pipeline(
                    masks, positions, 1.8, COLOR, pipeline
                )
                second = render_with_pipeline(
                    masks, positions, 1.8, COLOR, pipeline
                )
                self.assertEqual(first.image.tobytes(), second.image.tobytes())
                margin = math.ceil((pipeline.outline_pixels(1.8) + 1) * 1.8)
                self.assertEqual(first.text_origin, (margin, margin))
                self.assertEqual(
                    first.image.size,
                    (
                        math.ceil(composed.width * 1.8) + 2 * margin,
                        math.ceil(composed.height * 1.8) + 2 * margin,
                    ),
                )
                self.assertEqual(first.image.mode, "RGBA")

    def test_outline_after_scaling_keeps_a_crisp_black_ring(self):
        masks, positions = synthetic_layout()
        pipeline = ScalePipeline(outline_stage="after", resample="nearest")
        rendered = render_with_pipeline(masks, positions, 2, COLOR, pipeline)
        alpha = np.asarray(rendered.image)[..., 3]
        fill = np.asarray(rendered.image)[..., 1] == COLOR[1]
        expected_ring = (
            Image.fromarray(fill.astype(np.uint8) * 255)
            .filter(ImageFilter.MaxFilter(3))
            .point(lambda v: 255 if v else 0)
        )
        self.assertTrue(np.array_equal(alpha, np.asarray(expected_ring)))

    def test_without_outline_the_alpha_is_the_scaled_ink_alone(self):
        masks, positions = synthetic_layout()
        composed = _compose(masks, positions)
        rendered = render_with_pipeline(
            masks, positions, 1, COLOR, DEFAULT_PIPELINE, outlined=False
        )
        self.assertEqual(rendered.text_origin, (1, 1))
        self.assertEqual(
            rendered.image.size, (composed.width + 2, composed.height + 2)
        )
        pixels = np.asarray(rendered.image)
        expected = np.zeros(pixels.shape[:2], dtype=np.uint8)
        expected[1:-1, 1:-1] = np.asarray(composed)
        self.assertTrue(np.array_equal(pixels[..., 3], expected))
        self.assertTrue((pixels[expected > 0, :3] == COLOR).all())

    def test_without_outline_the_outline_width_changes_nothing(self):
        # The stage still decides whether color is applied before or
        # after scaling, so only the width axis collapses.
        masks, positions = synthetic_layout()
        for stage in ("before", "after"):
            with self.subTest(stage=stage):
                renders = {
                    render_with_pipeline(
                        masks,
                        positions,
                        1.8,
                        COLOR,
                        ScalePipeline(outline_stage=stage, outline_px=width),
                        outlined=False,
                    ).image.tobytes()
                    for width in ("1", "round")
                }
                self.assertEqual(len(renders), 1)
                outlined = render_with_pipeline(
                    masks,
                    positions,
                    1.8,
                    COLOR,
                    ScalePipeline(outline_stage=stage),
                )
                self.assertNotIn(outlined.image.tobytes(), renders)

    def test_glyph_rounding_changes_only_pixels_after_the_first_glyph(self):
        masks, positions = synthetic_layout()
        continuous = render_with_pipeline(
            masks, positions, 1.3, COLOR, ScalePipeline(resample="nearest")
        )
        floored = render_with_pipeline(
            masks,
            positions,
            1.3,
            COLOR,
            ScalePipeline(resample="nearest", glyph_x_rounding="floor"),
        )
        # The first glyph sits at native x 0, so rounding only moves the
        # second glyph, which starts after column 3 + 6 * 1.3.
        left = np.asarray(continuous.image)[:, :11]
        self.assertTrue(
            np.array_equal(left, np.asarray(floored.image)[:, :11])
        )
        self.assertNotEqual(
            continuous.image.tobytes(), floored.image.tobytes()
        )


class OtsuPortTests(unittest.TestCase):
    def test_matches_vectors_recorded_from_nickname_pixels_ts(self):
        # Vectors were produced by dfragon's nickname-pixels.ts from
        # synthetic arrays; they hold no screenshot pixels. Each vector
        # records the output of one or more function chains.
        fixture = json.loads(
            (FIXTURES / "otsu-vectors.json").read_text(encoding="utf-8")
        )
        recorded = {
            chain
            for vector in fixture["vectors"]
            for chain in vector["outputs"]
        }
        self.assertEqual(recorded, set(PORTED_CHAINS))
        for vector in fixture["vectors"]:
            rgba = np.array(vector["rgba"], dtype=np.uint8).reshape(
                vector["height"], vector["width"], 4
            )
            for chain, expected in vector["outputs"].items():
                with self.subTest(name=vector["name"], chain=chain):
                    output = rgba
                    for step in PORTED_CHAINS[chain]:
                        output = step(output)
                    self.assertEqual(output.reshape(-1).tolist(), expected)

    def test_raid_vectors_exercise_line_and_detached_removal(self):
        # The fixture must keep a vector where each raid stage changes
        # the result, so the port of that stage is actually checked.
        fixture = json.loads(
            (FIXTURES / "otsu-vectors.json").read_text(encoding="utf-8")
        )
        line_removed = detached_removed = False
        for vector in fixture["vectors"]:
            outputs = vector["outputs"]
            text = outputs.get("binarizeRaidTextPixels")
            lines = outputs.get(
                "binarizeRaidTextPixels,removeRaidNicknameFrameLines"
            )
            final = outputs.get("binarizeRaidNicknamePixels")
            line_removed |= text is not None and lines not in (None, text)
            detached_removed |= lines is not None and final not in (
                None,
                lines,
            )
        self.assertTrue(line_removed)
        self.assertTrue(detached_removed)

    def test_gray_uses_the_product_integer_weights(self):
        self.assertEqual(int(bt601_gray(np.array([255, 255, 255]))), 255)
        self.assertEqual(int(bt601_gray(np.array([75, 209, 255]))), 174)
        self.assertEqual(int(bt601_gray(np.array([1, 0, 0]))), 0)
        self.assertEqual(int(bt601_gray(np.array([0, 1, 0]))), 1)

    def test_threshold_splits_two_well_separated_levels(self):
        histogram = [0] * 256
        histogram[40] = 10
        histogram[200] = 10
        threshold = otsu_threshold(histogram, 20, 40 * 10 + 200 * 10)
        self.assertGreaterEqual(threshold, 40)
        self.assertLess(threshold, 200)

    def test_empty_input_returns_empty(self):
        self.assertEqual(
            binarize_nickname_pixels(np.zeros((0, 4), dtype=np.uint8)).size, 0
        )


class RaidBinarizationTests(unittest.TestCase):
    def test_darkest_channel_keeps_text_the_luma_would_merge_with_glare(self):
        # Yellow glare (255, 230, 40) is brighter than pale text
        # (190, 190, 150) in BT.601 luma but darker in its blue channel.
        crop = np.zeros((2, 4, 4), dtype=np.uint8)
        crop[..., 3] = 255
        crop[..., :3] = (60, 40, 10)
        crop[0, 1, :3] = (255, 230, 40)
        crop[1, 2, :3] = (190, 190, 150)
        party_ink = binarize_nickname_pixels(crop)[..., 0] == 0
        raid_ink = binarize_raid_text_pixels(crop)[..., 0] == 0
        self.assertTrue(party_ink[0, 1])
        self.assertFalse(raid_ink[0, 1])
        self.assertTrue(raid_ink[1, 2])
        self.assertEqual(int(raid_ink.sum()), 1)

    def test_frame_lines_are_cleared_only_in_edge_rows(self):
        # Height 10 has a 2-row margin; a line needs 15 of 20 columns.
        crop = binary_crop(
            [
                "..###############...",
                "....................",
                "....................",
                "..##................",
                "####################",
                "..##................",
                "....................",
                "....................",
                "...##########.......",
                "###############.....",
            ]
        )
        cleared = remove_raid_nickname_frame_lines(crop)
        self.assertEqual(
            ink_rows(cleared),
            [
                "....................",
                "....................",
                "....................",
                "..##................",
                "####################",
                "..##................",
                "....................",
                "....................",
                "...##########.......",
                "....................",
            ],
        )
        self.assertTrue(np.array_equal(cleared[..., 3], crop[..., 3]))

    def test_detached_clusters_far_from_the_largest_are_cleared(self):
        # Height 17 clears clusters at least 4 empty columns away from
        # the cluster with the most ink; ties keep the leftmost.
        cases = {
            "##...#": "##...#",
            "##....#": "##.....",
            "#....##": ".....##",
            "#....#": "#.....",
            "#....#....###": "..........###",
        }
        for pattern, expected in cases.items():
            with self.subTest(pattern=pattern):
                cleared = remove_detached_raid_nickname_pixels(
                    binary_crop([pattern] * 17)
                )
                self.assertEqual(ink_rows(cleared), [expected] * 17)

    def test_raid_binarization_needs_a_two_dimensional_crop(self):
        with self.assertRaises(ValueError):
            binarize_raid_nickname_pixels(np.zeros((8, 4), dtype=np.uint8))


class ReferenceScreenTests(unittest.TestCase):
    def setUp(self):
        # A 20x10 crop with a bright frame line in row 0 and a block of
        # text; both are ink to the party Otsu, only the text to raid.
        self.crop = np.zeros((10, 20, 4), dtype=np.uint8)
        self.crop[..., 3] = 255
        self.crop[..., :3] = (30, 40, 60)
        self.crop[0, 1:19, :3] = COLOR
        self.crop[3:8, 4:9, :3] = COLOR

    def test_party_reference_keeps_the_frame_line_as_ink(self):
        reference = Reference.from_rgba(self.crop, 1.0, "party")
        self.assertTrue(reference.ink[0, 1:19].all())
        self.assertEqual(reference.ink_count, 18 + 25)
        self.assertEqual(reference.screen, "party")

    def test_raid_reference_drops_the_frame_line(self):
        reference = Reference.from_rgba(self.crop, 1.0, "raid")
        self.assertFalse(reference.ink[0].any())
        self.assertTrue(reference.ink[3:8, 4:9].all())
        self.assertEqual(reference.ink_count, 25)
        self.assertEqual(reference.screen, "raid")

    def test_comparison_binarizes_like_the_reference_screen(self):
        # The synthetic repaints the frame line; party counts the whole
        # repainted line as unchanged ink, raid ignores it either way.
        synthetic = np.zeros((1, 18, 4), dtype=np.uint8)
        synthetic[..., :3] = COLOR
        synthetic[..., 3] = 255
        for screen in ("party", "raid"):
            with self.subTest(screen=screen):
                reference = Reference.from_rgba(self.crop, 1.0, screen)
                self.assertEqual(
                    compare(reference, synthetic, (1, 0)).mismatch, 0
                )
                self.assertEqual(
                    compare(reference, synthetic, (1, 1)).mismatch,
                    18 if screen == "party" else 0,
                )

    def test_unknown_screen_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "screen"):
            Reference.from_rgba(self.crop, 1.0, "hud")


class AlignmentTests(unittest.TestCase):
    def setUp(self):
        masks, positions = synthetic_layout()
        self.rendered = render_with_pipeline(
            masks, positions, 1.8, COLOR, DEFAULT_PIPELINE
        )
        self.synthetic = np.asarray(self.rendered.image)
        self.background = band_background(80, 30)

    def test_recovers_a_known_placement_with_zero_residual(self):
        for known in ((7, 4), (12, 9), (0, 0)):
            with self.subTest(origin=known):
                crop = place(self.background, self.rendered.image, known)
                reference = Reference.from_rgba(crop, 1.8, "party")
                estimated = estimate_text_origin(
                    reference,
                    self.synthetic,
                    self.rendered.text_origin,
                    COLOR,
                )
                # Perturb the base so the search must undo the shift.
                base = (estimated[0] + 2, estimated[1] - 1)
                best = best_candidate(
                    search_alignment(
                        reference,
                        self.synthetic,
                        self.rendered.text_origin,
                        base,
                    )
                )
                self.assertEqual(best.origin, known)
                self.assertEqual(best.offset, (-2, 1))
                self.assertEqual(best.comparison.mismatch, 0)
                self.assertEqual(best.comparison.mae, 0.0)
                self.assertEqual(best.comparison.ink, reference.ink_count)

    def test_search_covers_the_full_square_of_offsets(self):
        crop = place(self.background, self.rendered.image, (5, 5))
        candidates = search_alignment(
            Reference.from_rgba(crop, 1.8, "party"),
            self.synthetic,
            self.rendered.text_origin,
            (
                5 + self.rendered.text_origin[0],
                5 + self.rendered.text_origin[1],
            ),
            radius=2,
        )
        self.assertEqual(len(candidates), 25)
        self.assertEqual(
            {candidate.offset for candidate in candidates},
            {(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)},
        )

    def test_shifted_placement_scores_worse_than_the_true_one(self):
        crop = place(self.background, self.rendered.image, (5, 5))
        reference = Reference.from_rgba(crop, 1.8, "party")
        exact = compare(reference, self.synthetic, (5, 5))
        shifted = compare(reference, self.synthetic, (6, 5))
        self.assertEqual(exact.mismatch, 0)
        self.assertGreater(shifted.mismatch, 0)
        self.assertGreater(shifted.mae, exact.mae)

    def test_placement_outside_the_crop_has_infinite_mae(self):
        reference = Reference.from_rgba(self.background, 1.8, "party")
        self.assertTrue(
            math.isinf(compare(reference, self.synthetic, (200, 200)).mae)
        )


class EdgeProfileTests(unittest.TestCase):
    def test_crisp_edge_rises_within_one_row(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[6:9, :, 0] = 220
        profile = edge_profile(image, 2, 16, 6, window=3)
        self.assertEqual(profile.rows, list(range(3, 10)))
        self.assertAlmostEqual(profile.rise_rows, 0.8, places=6)

    def test_ramped_edge_spreads_over_the_ramp(self):
        image = np.zeros((12, 20, 3), dtype=np.uint8)
        image[5, :, 0] = 55
        image[6, :, 0] = 110
        image[7, :, 0] = 165
        image[8:, :, 0] = 220
        profile = edge_profile(image, 0, 20, 6, window=3)
        self.assertAlmostEqual(profile.rise_rows, 3.2, places=6)

    def test_flat_profile_has_no_rise(self):
        image = np.full((10, 10, 3), 50, dtype=np.uint8)
        self.assertTrue(math.isnan(edge_profile(image, 0, 10, 5).rise_rows))

    def test_window_outside_the_image_is_an_error(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            edge_profile(image, 0, 4, 1, window=3)


class LineProfileTests(unittest.TestCase):
    def test_crisp_two_row_line_measures_two_rows(self):
        # Row means of an s=1 raid separator: two black rows with a
        # brighter row under them than above.
        self.assertAlmostEqual(
            line_width([10, 29, 0, 0, 37, 21, 22]), 2.0, places=6
        )

    def test_scaled_line_measures_its_fractional_width(self):
        # Half levels: 10 above (between 5 and 20) and 15 below
        # (between 10 and 30), so 1 + 5/15 + 2 + 5/20 rows.
        self.assertAlmostEqual(
            line_width([20, 5, 0, 0, 10, 30, 20]), 3.5833333, places=6
        )

    def test_flat_profile_has_no_width(self):
        self.assertTrue(math.isnan(line_width([5, 5, 5])))

    def test_line_center_carries_the_sub_pixel_phase(self):
        # A symmetric two-row line centers between its rows; a line
        # leaking a third of its depth into the row below sits lower.
        self.assertAlmostEqual(
            line_center([20, 20, 0, 0, 20, 20, 20]), 2.5, places=6
        )
        self.assertGreater(line_center([20, 20, 0, 0, 10, 20, 20]), 2.5)
        self.assertTrue(math.isnan(line_center([5, 5, 5])))

    def test_line_profile_averages_the_columns_of_the_band(self):
        image = np.full((12, 20, 1), 40, dtype=np.uint8)
        image[5:7, :, 0] = 0
        image[5:7, 10:, 0] = 20
        profile = line_profile(image, 0, 20, 5, window=3)
        self.assertEqual(profile.rows, list(range(2, 9)))
        self.assertEqual(profile.values[3:5], [10.0, 10.0])
        self.assertAlmostEqual(profile.width_rows, 2.0, places=6)


class RaidSeparatorTests(unittest.TestCase):
    def test_measures_the_lines_under_the_name_column_of_each_case(self):
        anchor = (100, 50)
        image = np.full((400, 600, 3), 20, dtype=np.uint8)
        for line in range(1, 12):
            top = anchor[1] + 18 + 21 * line
            image[top : top + 2] = 0
            image[top + 2] = 40
        with tempfile.TemporaryDirectory() as directory:
            Image.fromarray(image, "RGB").save(Path(directory) / "case.png")
            index = {
                "crops": [
                    {
                        "caseId": "case",
                        "slot": 1,
                        "scale": 1,
                        "crop": {"x": anchor[0] + 182, "y": anchor[1] + 21},
                    },
                    {"caseId": "case", "slot": 2},
                ]
            }
            results = raid_separator_profiles(index, Path(directory), 3)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["caseId"], "case")
        self.assertEqual(len(results[0]["lines"]), 11)
        self.assertEqual(results[0]["lines"][0]["values"][3:5], [0.0, 0.0])
        self.assertAlmostEqual(results[0]["width_rows"], 2.0, places=6)
        # Line 5 covers rows anchor + 18 + 105 and the next one.
        self.assertAlmostEqual(
            results[0]["lines"][4]["center_row"], 50 + 123 + 0.5, places=6
        )
        self.assertEqual(line_width_fit(results), {"lines": 0})

    def test_width_fit_reports_the_observed_line_through_scaled_lines(self):
        results = [
            {"scale": 1.0, "lines": [{"width_rows": 2.0}]},
            {
                "scale": 1.5,
                "lines": [{"width_rows": 2.8}, {"width_rows": 2.8}],
            },
            {
                "scale": 1.8,
                "lines": [{"width_rows": 3.4}, {"width_rows": math.nan}],
            },
        ]
        fit = line_width_fit(results)
        self.assertEqual(fit["lines"], 3)
        self.assertAlmostEqual(fit["slope"], 2.0, places=6)
        self.assertAlmostEqual(fit["intercept"], -0.2, places=6)
        self.assertAlmostEqual(fit["correlation"], 1.0, places=6)


def no_symbols(character):
    raise AssertionError(f"unexpected symbol raster for {character!r}")


class LayoutTests(unittest.TestCase):
    def glyphs(self, names, widths, positions, advances):
        masks = [Image.new("L", (width, 10), 255) for width in widths]

        return GlyphLayout(names, masks, positions, advances)

    def test_hangul_glyphs_start_at_the_cursor_and_advance_by_ink(self):
        # Font advance 11 with bearings 1, 0, 1; the raid window packs
        # each glyph at ink width + 1 instead.
        glyphs = self.glyphs(
            ["dotum"] * 3,
            [8, 9, 10],
            [(1, -9), (11, -9), (23, -9)],
            [11, 11, 11],
        )
        spaced = ink_advance_layout("가나다", glyphs, no_symbols)
        self.assertEqual(spaced.positions, [(0, -9), (9, -9), (19, -9)])
        self.assertEqual(spaced.advances, [9, 10, 11])
        self.assertEqual(spaced.masks, glyphs.masks)
        self.assertEqual(spaced.names, glyphs.names)

    def test_other_faces_and_non_hangul_dotum_keep_their_bearing(self):
        # Dotumche "D" and a Dotum kana keep the renderer's placement
        # relative to the cursor; only the Hangul glyphs re-pack.
        glyphs = self.glyphs(
            ["dotum", "dotumche", "dotum", "dotum"],
            [8, 5, 9, 9],
            [(1, -9), (11, -7), (18, -9), (30, -9)],
            [11, 6, 11, 11],
        )
        spaced = ink_advance_layout("가Dぎ나", glyphs, no_symbols)
        self.assertEqual(
            spaced.positions, [(0, -9), (9, -7), (16, -9), (26, -9)]
        )
        self.assertEqual(spaced.advances, [9, 6, 11, 10])

    def test_ascii_symbols_take_the_dotumche_glyph_at_the_cursor(self):
        # The renderer gives "!" the 1x8 Dotum glyph one px right of
        # the cursor and advance 3; the raid window draws Dotumche's
        # "!" at the cursor and advances its ink width + 1.
        glyphs = self.glyphs(
            ["dotum", "dotum"], [8, 1], [(1, -9), (10, -8)], [11, 3]
        )
        dotumche = Image.new("L", (2, 8), 255)

        def symbol_raster(character):
            self.assertEqual(character, "!")
            return dotumche, (3, -7)

        spaced = ink_advance_layout("가!", glyphs, symbol_raster)
        self.assertEqual(spaced.names, ["dotum", "dotumche"])
        self.assertIs(spaced.masks[1], dotumche)
        self.assertEqual(spaced.positions, [(0, -9), (9, -7)])
        self.assertEqual(spaced.advances, [9, 3])

    def test_advance_option_accepts_ink_and_font(self):
        required = ["--crops", "i.json", "--manifest", "m.json", "--out", "o"]
        self.assertEqual(_parse_measured(required).advance, "ink")
        self.assertEqual(
            _parse_measured([*required, "--advance", "font"]).advance, "font"
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                _parse_measured([*required, "--advance", "metrics"])


@unittest.skipUnless(GULIM.is_file(), "Supply the local Windows gulim font")
class RaidLayoutTests(unittest.TestCase):
    def test_ink_mode_draws_symbols_with_dotumche_and_font_mode_does_not(self):
        renderer = Renderer(FontPaths(gulim=GULIM))
        # Read off s=1 crops: the game's "!" has a two-row dot and the
        # "-" sits one row lower than Dotum's, both as Dotumche draws
        # them, and each starts at the cursor with ink width + 1.
        ink = raid_layout(renderer, "앞!-", "dotum", "ink")
        self.assertEqual(ink.names, ["dotum", "dotumche", "dotumche"])
        self.assertEqual(ink.positions[1], (ink.advances[0], -8))
        self.assertEqual(ink.advances[1], 2)
        self.assertEqual(
            ink_rows(np.asarray(ink.masks[1])[..., None].repeat(4, 2) ^ 255),
            ["#", "#", "#", "#", "#", ".", "#", "#"],
        )
        self.assertEqual(ink.positions[2], (ink.advances[0] + 2, -4))
        font = raid_layout(renderer, "앞!-", "dotum", "font")
        self.assertEqual(font.names, ["dotum", "dotum", "dotum"])
        self.assertEqual(font.positions, renderer.layout("앞!-").positions)
        with self.assertRaises(ValueError):
            raid_layout(renderer, "앞", "dotum", "metrics")


def trapezoid_taps():
    # Flat to 0.5 crop px, then linear to zero at 1.5 crop px.
    values = tuple(
        max(0.0, min(1.0, (1.5 - distance) / 1.0))
        for distance in (index * 0.25 for index in range(tap_count("crop")))
    )

    return KernelTaps(values, values)


def kernel_crop(glyphs, scale, origin, taps, size, fill_value=146, bg=20):
    fill, alpha = native_layers(glyphs)
    coverage, filled = resample(fill, alpha, scale, origin, taps, size)
    darkest = np.clip(bg * (1 - coverage) + fill_value * filled, 0, 255)
    rgba = np.zeros((*size, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(darkest)[..., None]
    rgba[..., 3] = 255

    return rgba


class KernelTests(unittest.TestCase):
    def test_tap_basis_interpolates_between_neighboring_taps(self):
        basis = tap_basis(np.array([0.1, -0.1, 0.25, 4.5]), 0.25, 17)
        self.assertAlmostEqual(basis[0, 0], 0.6)
        self.assertAlmostEqual(basis[0, 1], 0.4)
        np.testing.assert_allclose(basis[1], basis[0])
        self.assertAlmostEqual(basis[2, 1], 1.0)
        self.assertEqual(basis[3].sum(), 0.0)
        distances = np.linspace(-3, 3, 25)
        np.testing.assert_allclose(
            tent_taps(2.0).weights(distances, 2.0, "x"),
            np.maximum(0, 1 - np.abs(distances) / 2.0),
        )

    def test_native_taps_describe_the_kernel_in_source_pixels(self):
        native = tent_taps(1.5, "native")
        self.assertEqual(native.unit, "native")
        self.assertEqual(native.spacing, 0.125)
        distances = np.linspace(-3, 3, 25)
        for scale in (1.2, 1.8):
            with self.subTest(scale=scale):
                np.testing.assert_allclose(
                    native.weights(distances, scale, "y"),
                    np.maximum(0, 1 - np.abs(distances) / scale),
                )
                self.assertAlmostEqual(native.integral("x", scale), 1.0)
        with self.assertRaises(ValueError):
            KernelTaps((1.0,), (1.0,), "texel")

    def test_integral_counts_the_last_tap_once(self):
        # Three taps of 1 at 0, 0.25 and 0.5 crop px make a box from
        # -0.5 to 0.5, so the area is 1 crop px, not 1.25.
        box = KernelTaps((1.0, 1.0, 1.0), (1.0, 0.0, 0.0))
        self.assertAlmostEqual(box.integral("x", 1.0), 1.0)
        self.assertAlmostEqual(box.integral("x", 2.0), 0.5)
        self.assertAlmostEqual(box.integral("y", 1.0), 0.25)
        unit = box.unit_gain(1.0)
        self.assertAlmostEqual(
            unit.integral("x", 1.0) * unit.integral("y", 1.0), 1.0
        )
        self.assertEqual(unit.y, box.y)

    def test_standard_shapes_are_unit_area_and_the_gain_scales_x(self):
        distances = np.linspace(-4, 4, 81)
        for shape in ("nearest", "bilinear", "catmull-rom"):
            with self.subTest(shape=shape):
                kernel = StandardKernel(shape)
                self.assertAlmostEqual(kernel.integral("x", 1.5), 1.0, 3)
                self.assertAlmostEqual(kernel.integral("y", 1.5), 1.0, 3)
        doubled = StandardKernel("nearest2x-bilinear")
        np.testing.assert_allclose(
            doubled.weights(np.array([0.0, 0.45, 0.9, 1.35]), 1.8, "y"),
            [1.0, 1.0, 0.5, 0.0],
        )
        np.testing.assert_allclose(
            StandardKernel("box-bilinear").weights(
                np.array([0.0, 1.0, 1.5]), 1.0, "y"
            ),
            [0.75, 0.125, 0.0],
        )
        gained = StandardKernel("bilinear", 1.1)
        np.testing.assert_allclose(
            gained.weights(distances, 1.0, "x"),
            1.1 * np.maximum(0, 1 - np.abs(distances)),
        )
        self.assertAlmostEqual(gained.integral("x", 1.0), 1.1, 3)
        self.assertAlmostEqual(gained.integral("y", 1.0), 1.0, 3)
        with self.assertRaises(ValueError):
            StandardKernel("lanczos")

    def test_native_layers_pad_by_two_and_outline_by_one(self):
        mask = Image.new("L", (1, 2), 255)
        fill, alpha = native_layers(
            GlyphLayout(["dotum"], [mask], [(0, 0)], [2])
        )
        self.assertEqual(fill.shape, (6, 5))
        self.assertEqual(
            ink_rows(np.where(fill, 0, 255)[..., None].repeat(4, 2)),
            [".....", ".....", "..#..", "..#..", ".....", "....."],
        )
        self.assertEqual(
            ink_rows(np.where(alpha, 0, 255)[..., None].repeat(4, 2)),
            [".....", ".###.", ".###.", ".###.", ".###.", "....."],
        )

    def test_box_taps_double_a_pixel_at_scale_two(self):
        box = KernelTaps(
            (1.0, 1.0, 1.0, 1.0) + (0.0,) * (tap_count("crop") - 4), ()
        )
        fill = np.ones((1, 1), dtype=bool)
        coverage, filled = resample(
            fill, fill, 2.0, (1.0, 1.0), KernelTaps(box.x, box.x), (4, 4)
        )
        expected = np.zeros((4, 4))
        expected[1:3, 1:3] = 1
        np.testing.assert_allclose(coverage, expected)
        np.testing.assert_allclose(filled, expected)

    def test_model_composite_blends_coverage_over_the_background(self):
        crop = np.full((1, 4, 4), 255, dtype=np.uint8)
        crop[0, :, :3] = 20
        reference = Reference.from_rgba(crop, 1.0, "raid")
        coverage = np.array([[0.0, 0.5, 1.0, 1.1]])
        filled = np.array([[0.0, 0.25, 1.0, 1.1]])
        composite, covered = model_composite(
            reference, coverage, filled, (200, 100, 50)
        )
        self.assertEqual(covered.tolist(), [[False, True, True, True]])
        self.assertEqual(composite[0, 0, :3].tolist(), [20, 20, 20])
        self.assertEqual(composite[0, 1, :3].tolist(), [60, 35, 22])
        self.assertEqual(composite[0, 2, :3].tolist(), [200, 100, 50])
        # Overshoot is kept, not clipped to the fill color.
        self.assertEqual(composite[0, 3, :3].tolist(), [218, 108, 53])

    def test_alignment_and_estimation_recover_a_known_kernel(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        taps = trapezoid_taps()
        scale = 1.8
        crops = [
            kernel_crop(glyphs, scale, origin, taps, (34, 48))
            for origin in ((3.3, 2.1), (7.0, 4.6))
        ]
        references = [
            Reference.from_rgba(crop, scale, "raid") for crop in crops
        ]
        # The bilinear start misplaces by a tenth of a pixel, and the
        # second pass with the first estimate settles the placement.
        estimated = None
        for _ in range(2):
            samples = [
                align_sample(reference, glyphs, 146, scale, estimated)
                for reference in references
            ]
            estimated = estimate_kernel(samples, scale)
        for sample, origin in zip(samples, ((3.3, 2.1), (7.0, 4.6))):
            self.assertAlmostEqual(sample.origin[0], origin[0], delta=0.03)
            self.assertAlmostEqual(sample.origin[1], origin[1], delta=0.03)
        for sample in samples:
            modeled, _ = sample.model(estimated)
            crop_dark = sample.reference.rgba[..., 0].astype(float)
            self.assertLess(np.abs(modeled - crop_dark).max(), 4.0)
        # The axis gains trade off against each other, so compare the
        # geometric mean of the x and y taps where the kernel is wide.
        recovered = np.sqrt(
            np.array(estimated.x[:7]) * np.array(estimated.y[:7])
        )
        np.testing.assert_allclose(recovered, taps.x[:7], atol=0.05)
        # A separable fit fixes only the product of the axis gains, and
        # the trapezoid integrates to 2 crop px, 1.111 source px.
        self.assertAlmostEqual(
            estimated.integral("x", scale) * estimated.integral("y", scale),
            1.111**2,
            delta=0.05,
        )

    def test_native_estimation_recovers_one_kernel_across_scales(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        # The same source-pixel kernel, a tent to 0.75 px, rendered at
        # two scales; the native fit must find it from both at once.
        truth = StandardKernel("nearest2x-bilinear")
        samples = []
        for scale, origin in ((1.4, (3.3, 2.1)), (1.8, (5.0, 4.6))):
            crop = kernel_crop(glyphs, scale, origin, truth, (34, 48))
            reference = Reference.from_rgba(crop, scale, "raid")
            samples.append(align_sample(reference, glyphs, 146, scale, truth))
        estimated = estimate_kernel(samples, 1.6, unit="native")
        self.assertEqual(estimated.unit, "native")
        # The axis gains trade off against each other, so compare the
        # product of the x and y taps out to where the tent ends.
        product = np.array(estimated.x[:8]) * np.array(estimated.y[:8])
        expected = np.clip(1.5 - 2 * np.array(estimated.distances[:8]), 0, 1)
        np.testing.assert_allclose(product, expected**2, atol=0.05)

    def test_gain_fit_recovers_a_brightened_bilinear(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        truth = StandardKernel("bilinear", 1.12)
        crop = kernel_crop(glyphs, 1.5, (3.0, 2.0), truth, (30, 40))
        reference = Reference.from_rgba(crop, 1.5, "raid")
        sample = align_sample(reference, glyphs, 146, 1.5, truth)
        self.assertAlmostEqual(
            fit_gain([sample], StandardKernel("bilinear")), 1.12, delta=0.01
        )
        with self.assertRaises(ValueError):
            fit_gain([], StandardKernel("bilinear"))

    def test_layers_larger_than_the_crop_are_rejected(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        tiny = Reference.from_rgba(
            np.full((8, 8, 4), 255, dtype=np.uint8), 1.0, "raid"
        )
        with self.assertRaises(ValueError):
            align_sample(tiny, glyphs, 146, 1.0)


def measured_settings(**overrides):
    settings = {
        "candidates": ["measured-bucket"],
        "holdouts": [],
        "residuals": ["measured-bucket"],
        "iterations": 2,
        "scale_source": "detected",
        "workers": 1,
    }
    settings.update(overrides)

    return settings


class MeasuredKernelTests(unittest.TestCase):
    def sample_info(self, sample_id, scale, case_id="c", raid="A"):
        return SampleInfo(
            sample_id,
            case_id,
            "raid",
            1,
            f"{sample_id}.png",
            scale,
            "가",
            0,
            "",
            raid,
            scale,
        )

    def run_measured(self, infos, references, settings):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        skipped = []
        with tempfile.TemporaryDirectory() as directory:
            run_measured_kernel(
                Path(directory),
                infos,
                skipped,
                lambda sample: (glyphs, *references[sample.id]),
                settings,
            )
            payload = json.loads(
                (Path(directory) / "measured-kernel.json").read_text("utf-8")
            )
            report = (Path(directory) / "measured-kernel.md").read_text(
                "utf-8"
            )
            residuals = sorted(
                path.name for path in (Path(directory) / "residual").iterdir()
            )

        return payload, report, residuals, skipped

    def references(self, glyphs, taps, placements):
        return {
            name: (
                Reference.from_rgba(
                    kernel_crop(glyphs, scale, origin, taps, (34, 48)),
                    scale,
                    "raid",
                ),
                color,
            )
            for name, (scale, origin, color) in placements.items()
        }

    def test_bucket_without_plain_rows_is_left_unevaluated(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        references = self.references(
            glyphs,
            trapezoid_taps(),
            {
                "y1": (1.5, (3.0, 2.0), GLOW_FILL),
                "b1": (1.8, (3.3, 2.1), PLAIN_FILL),
            },
        )
        payload, report, _, skipped = self.run_measured(
            [self.sample_info("y1", 1.5), self.sample_info("b1", 1.8)],
            references,
            measured_settings(),
        )
        self.assertEqual(skipped, [])
        (measured,) = payload["candidates"]
        in_sample = measured["evaluations"]["in_sample"]
        self.assertEqual(in_sample["unevaluated"], ["y1"])
        self.assertEqual(list(in_sample["groups"]), ["1.8"])
        self.assertIn("빠진 crop: y1", report)

    def test_plain_rows_fit_the_bucket_and_every_row_is_scored(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        # Two plain rows at different sub-pixel phases pin the kernel
        # down; the glow row shares a phase and is only scored, and the
        # s=1 crop is scored by the fixed shape alone.
        references = self.references(
            glyphs,
            trapezoid_taps(),
            {
                "b1": (1.8, (3.3, 2.1), PLAIN_FILL),
                "b2": (1.8, (7.0, 4.6), PLAIN_FILL),
                "y1": (1.8, (3.3, 2.1), GLOW_FILL),
            },
        )
        references.update(
            self.references(
                glyphs,
                StandardKernel("nearest"),
                {"u1": (1.0, (3.0, 2.0), PLAIN_FILL)},
            )
        )
        infos = [
            self.sample_info("b1", 1.8, "c1"),
            self.sample_info("b2", 1.8, "c2"),
            self.sample_info("y1", 1.8, "c1"),
            self.sample_info("u1", 1.0, "c3"),
        ]
        payload, report, residuals, skipped = self.run_measured(
            infos,
            references,
            measured_settings(
                candidates=["measured-bucket", "nearest"],
                holdouts=["capture", "raid"],
            ),
        )
        self.assertEqual(skipped, [])
        measured, nearest = payload["candidates"]
        self.assertEqual(measured["name"], "measured-bucket")
        in_sample = measured["evaluations"]["in_sample"]
        self.assertEqual(list(in_sample["groups"]), ["1.8"])
        self.assertEqual(in_sample["groups"]["1.8"]["plain_samples"], 2)
        self.assertEqual(in_sample["groups"]["1.8"]["samples"], 3)
        self.assertEqual(in_sample["groups"]["1.8"]["mismatch"], 0)
        self.assertEqual(in_sample["groups"]["1.8"]["passed"], 3)
        self.assertEqual(
            [(fit["sample"], fit["plain"]) for fit in in_sample["fits"]],
            [("b1", True), ("b2", True), ("y1", False)],
        )
        self.assertEqual(in_sample["kernels"]["1.8"]["unit"], "crop")
        # Leaving capture c1 out leaves b2 alone to fit c1's rows, and
        # the raid hold-out has no other raid to fit from.
        capture = measured["evaluations"]["holdout_capture"]
        self.assertEqual(
            sorted((fit["sample"], fit["fold"]) for fit in capture["fits"]),
            [("b1", "c1"), ("b2", "c2"), ("y1", "c1")],
        )
        self.assertEqual(sorted(capture["kernels"]), ["1.8/c1", "1.8/c2"])
        raid = measured["evaluations"]["holdout_raid"]
        self.assertEqual(raid["fits"], [])
        self.assertEqual(sorted(raid["unevaluated"]), ["b1", "b2", "y1"])
        self.assertEqual(
            list(nearest["evaluations"]["in_sample"]["groups"]), ["s=1", "1.8"]
        )
        self.assertEqual(
            nearest["evaluations"]["in_sample"]["groups"]["s=1"]["mismatch"],
            0,
        )
        self.assertNotIn("holdout_capture", nearest["evaluations"])
        self.assertEqual(
            payload["rows"],
            [{"raid": "A", "slot": 1, "text": "가", "plain": 3, "glow": 1}],
        )
        self.assertIn("| measured-bucket | 0/", report)
        self.assertIn("## s=1 구간", report)
        self.assertIn("## s 1.8 구간", report)
        self.assertEqual(
            residuals,
            [
                "b1_measured-bucket.png",
                "b2_measured-bucket.png",
                "y1_measured-bucket.png",
            ],
        )

    def test_derived_candidate_reuses_the_base_fit_and_gain_is_fitted(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        references = self.references(
            glyphs,
            StandardKernel("bilinear", 1.1),
            {
                "b1": (1.5, (3.3, 2.1), PLAIN_FILL),
                "b2": (1.5, (7.0, 4.6), PLAIN_FILL),
            },
        )
        infos = [
            self.sample_info("b1", 1.5, "c1"),
            self.sample_info("b2", 1.5, "c2"),
        ]
        payload, _, _, _ = self.run_measured(
            infos,
            references,
            measured_settings(
                candidates=[
                    "measured-bucket",
                    "measured-bucket-unit",
                    "bilinear-gain",
                ],
                holdouts=["capture"],
                residuals=[],
            ),
        )
        measured, unit, gain = payload["candidates"]
        base_kernel = measured["evaluations"]["in_sample"]["kernels"]["1.5"]
        unit_kernel = unit["evaluations"]["in_sample"]["kernels"]["1.5"]
        self.assertAlmostEqual(
            unit_kernel["integral_x"] * unit_kernel["integral_y"], 1.0
        )
        self.assertEqual(unit_kernel["taps_y"], base_kernel["taps_y"])
        self.assertEqual(
            sorted(unit["evaluations"]["holdout_capture"]["kernels"]),
            ["1.5/c1", "1.5/c2"],
        )
        self.assertAlmostEqual(
            gain["evaluations"]["in_sample"]["kernels"]["1.5"]["gain"],
            1.1,
            delta=0.02,
        )
        self.assertEqual(
            gain["evaluations"]["in_sample"]["groups"]["1.5"]["mismatch"], 0
        )

    def test_separate_fit_set_scores_every_crop_as_hold_out(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        references = self.references(
            glyphs,
            trapezoid_taps(),
            {
                "f1": (1.8, (3.3, 2.1), PLAIN_FILL),
                "f2": (1.8, (7.0, 4.6), PLAIN_FILL),
                "s1": (1.8, (5.0, 3.0), PLAIN_FILL),
                "s2": (1.3, (5.0, 3.0), PLAIN_FILL),
            },
        )
        skipped = []
        with tempfile.TemporaryDirectory() as directory:
            run_measured_kernel(
                Path(directory),
                [
                    self.sample_info("s1", 1.8, "d1", "B"),
                    self.sample_info("s2", 1.3, "d2", "B"),
                ],
                skipped,
                lambda sample: (glyphs, *references[sample.id]),
                measured_settings(holdouts=["capture"]),
                [
                    self.sample_info("f1", 1.8, "c1"),
                    self.sample_info("f2", 1.8, "c2"),
                ],
            )
            payload = json.loads(
                (Path(directory) / "measured-kernel.json").read_text("utf-8")
            )
            residuals = sorted(
                path.name for path in (Path(directory) / "residual").iterdir()
            )
        (measured,) = payload["candidates"]
        self.assertEqual(
            list(measured["evaluations"]), ["in_sample", "holdout_set"]
        )
        self.assertEqual(
            [
                fit["sample"]
                for fit in measured["evaluations"]["in_sample"]["fits"]
            ],
            ["f1", "f2"],
        )
        evaluation = measured["evaluations"]["holdout_set"]
        self.assertEqual(
            [(fit["sample"], fit["fold"]) for fit in evaluation["fits"]],
            [("s1", "1.8")],
        )
        self.assertEqual(evaluation["groups"]["1.8"]["mismatch"], 0)
        self.assertEqual(evaluation["unevaluated"], ["s2"])
        self.assertEqual(residuals, ["s1_measured-bucket.png"])

    def test_theory_scale_source_resamples_with_the_predicted_scale(self):
        masks, positions = synthetic_layout()
        glyphs = GlyphLayout(["dotum", "dotum"], masks, positions, [8, 5])
        reference, color = self.references(
            glyphs, trapezoid_taps(), {"b1": (1.8, (3.3, 2.1), PLAIN_FILL)}
        )["b1"]
        info = dataclasses.replace(self.sample_info("b1", 1.5), theory=1.8)
        payload, _, _, _ = self.run_measured(
            [info],
            {"b1": (reference, color)},
            measured_settings(
                candidates=["nearest2x-bilinear"],
                residuals=[],
                scale_source="theory",
            ),
        )
        fits = payload["candidates"][0]["evaluations"]["in_sample"]["fits"]
        self.assertEqual(fits[0]["group"], "1.5")
        self.assertAlmostEqual(fits[0]["origin"][0], 3.3, delta=0.1)


class FillColorTests(unittest.TestCase):
    def crop(self, bright, background):
        rgba = np.zeros((6, 12, 4), dtype=np.uint8)
        rgba[..., :3] = background
        rgba[..., 3] = 255
        rgba[2:4, 2:10, :3] = bright

        return rgba

    def test_glow_background_marks_the_row_whatever_its_fill_reads(self):
        # A scaled glow row whose blended fill mean sits nearer to the
        # plain fill than to the glow fill, over the golden background
        # the glow tints; the same fill over the neutral background is
        # a plain row.
        blended = (235, 224, 131)
        self.assertTrue(is_glow_row(self.crop(blended, (51, 42, 16))))
        self.assertEqual(
            raid_fill_color(self.crop(blended, (51, 42, 16))), GLOW_FILL
        )
        self.assertFalse(is_glow_row(self.crop(blended, (16, 16, 16))))
        self.assertEqual(
            raid_fill_color(self.crop(PLAIN_FILL, (15, 15, 15))), PLAIN_FILL
        )

    def test_crop_without_background_pixels_is_an_error(self):
        with self.assertRaises(ValueError):
            is_glow_row(self.crop((200, 200, 200), (120, 120, 120)))


class ReportTests(unittest.TestCase):
    def test_scale_buckets_follow_the_evaluation_script(self):
        self.assertEqual(scale_bucket(1.28), "1.2")
        self.assertEqual(scale_bucket(1.3), "1.3")
        self.assertEqual(scale_bucket(1.0), "1.0")
        self.assertEqual(scale_bucket(1.8), "1.8")

    def test_ranking_orders_by_mean_ratio_then_mae_per_bucket(self):
        samples = [
            SampleInfo("a", "a", "party", 1, "a.png", 1.0, "가", 0, "prtsc"),
            SampleInfo("b", "b", "raid", 1, "b.png", 1.82, "나", 100, "prtsc"),
        ]
        fits = [
            Fit("a", "p1", (0, 0), (0, 0), 3.0, 2, 100),
            Fit("a", "p2", (0, 0), (0, 0), 1.0, 2, 100),
            Fit("a", "p3", (0, 0), (0, 0), 0.5, 5, 100),
            Fit("b", "p1", (1, 0), (0, 0), 9.0, 0, 50),
            Fit("b", "p2", (0, 0), (0, 0), 1.0, 10, 50),
            Fit("b", "p3", (0, 0), (0, 0), 1.0, 1, 50),
        ]
        rankings = rank_pipelines(samples, fits, top=2)
        self.assertEqual(list(rankings), ["1.0", "1.8", "all"])
        self.assertEqual(
            [entry.pipeline for entry in rankings["1.0"]], ["p2", "p1"]
        )
        self.assertEqual(
            [entry.pipeline for entry in rankings["1.8"]], ["p1", "p3"]
        )
        self.assertEqual(
            [entry.pipeline for entry in rankings["all"]], ["p1", "p3"]
        )
        self.assertEqual(rankings["all"][0].mismatch, 2)
        self.assertEqual(rankings["all"][0].ink, 150)

    def test_report_files_and_residual_sheet_are_written(self):
        masks, positions = synthetic_layout()
        rendered = render_with_pipeline(
            masks, positions, 1.0, COLOR, DEFAULT_PIPELINE
        )
        crop = place(band_background(40, 20), rendered.image, (3, 2))
        sheet = residual_image(
            Reference.from_rgba(crop, 1.0, "party"),
            np.asarray(rendered.image),
            (3, 2),
        )
        self.assertEqual(sheet.size, (160, 242))
        samples = [
            SampleInfo("a", "a", "raid", 12, "a.png", 1.0, "가", 0, "prtsc")
        ]
        fits = [Fit("a", DEFAULT_PIPELINE.name, (0, 0), (3, 2), 0.0, 0, 10)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            write_report(
                out,
                samples,
                fits,
                [],
                rank_pipelines(samples, fits, 5),
                {"pipeline_count": 1, "radius": 2, "outline_px": 0},
            )
            payload = json.loads((out / "report.json").read_text("utf-8"))
            markdown = (out / "report.md").read_text("utf-8")
        self.assertEqual(payload["fits"][0]["ratio"], 0.0)
        self.assertEqual(payload["rankings"]["1.0"][0]["samples"], 1)
        self.assertEqual(payload["samples"][0]["screen"], "raid")
        self.assertIn(DEFAULT_PIPELINE.name, markdown)
        self.assertIn("| a | raid | 가 | 1.00 |", markdown)
        self.assertIn("렌더 외곽선 없음", markdown)


def crop_entry(case_id, screen, slot):
    return {
        "caseId": case_id,
        "screen": screen,
        "slot": slot,
        "file": f"{case_id}_slot{slot}.png",
        "scale": 1.29,
        "uiPercent": None,
        "capture": "ingame-jpg",
    }


class SampleLoadingTests(unittest.TestCase):
    def write_inputs(self, directory, index=None, manifest=None):
        crops = Path(directory) / "crops"
        crops.mkdir()
        if index is None:
            index = {
                "crops": [
                    crop_entry("c1", "party", 1),
                    crop_entry("c1", "party", 2),
                    crop_entry("c1", "party", 3),
                    crop_entry("r1", "raid", 12),
                ]
            }
        if manifest is None:
            manifest = {
                "cases": [
                    {
                        "id": "c1",
                        "screen": "party",
                        "slots": [
                            {"slot": 1, "text": "가나"},
                            {"slot": 2, "text": "다라", "uncertain": True},
                        ],
                    },
                    {
                        "id": "r1",
                        "screen": "raid",
                        "slots": [{"slot": 12, "text": "마바"}],
                    },
                ]
            }
        (crops / "index.json").write_text(json.dumps(index), "utf-8")
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")

        return crops / "index.json", path

    def test_joins_labels_and_skips_uncertain_or_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(directory)
            samples, skipped = load_samples(index, manifest, False)
            self.assertEqual(
                [(s.id, s.screen, s.slot, s.text, s.bucket) for s in samples],
                [
                    ("c1_slot1", "party", 1, "가나", "1.2"),
                    ("r1_slot12", "raid", 12, "마바", "1.2"),
                ],
            )
            self.assertEqual(
                [entry.sample for entry in skipped],
                ["c1_slot2", "c1_slot3"],
            )
            included, _ = load_samples(index, manifest, True)
            self.assertEqual(
                [s.id for s in included],
                ["c1_slot1", "c1_slot2", "r1_slot12"],
            )

    def test_missing_keys_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(directory)
            index.write_text('{"crops": [{"slot": 1}]}', "utf-8")
            with self.assertRaisesRegex(ValueError, "caseId"):
                load_samples(index, manifest, False)

    def test_index_without_screen_is_rejected_instead_of_read_as_party(self):
        entry = crop_entry("c1", "party", 1)
        del entry["screen"]
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(
                directory, index={"crops": [entry]}
            )
            with self.assertRaisesRegex(ValueError, r"crops\[0\].*screen"):
                load_samples(index, manifest, False)

    def test_manifest_case_without_screen_is_rejected(self):
        manifest = {
            "cases": [{"id": "c1", "slots": [{"slot": 1, "text": "가나"}]}]
        }
        with tempfile.TemporaryDirectory() as directory:
            index, path = self.write_inputs(directory, manifest=manifest)
            with self.assertRaisesRegex(ValueError, "manifest case.*screen"):
                load_samples(index, path, False)

    def test_unknown_and_mismatched_screens_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(
                directory, index={"crops": [crop_entry("c1", "hud", 1)]}
            )
            with self.assertRaisesRegex(ValueError, "party.*raid"):
                load_samples(index, manifest, False)
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(
                directory, index={"crops": [crop_entry("c1", "raid", 1)]}
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                load_samples(index, manifest, False)

    def test_outline_option_defaults_to_the_pipeline_outline(self):
        required = ["--crops", "i.json", "--manifest", "m.json", "--out", "o"]
        self.assertTrue(_parse_fit(required).outlined)
        self.assertFalse(_parse_fit([*required, "--outline-px", "0"]).outlined)

    def test_color_takes_a_triple_or_auto_and_advance_defaults_to_ink(self):
        required = ["--crops", "i.json", "--manifest", "m.json", "--out", "o"]
        self.assertEqual(_parse_fit(required).color, (75, 209, 255))
        self.assertEqual(_parse_fit(required).advance, "ink")
        self.assertIsNone(_parse_fit([*required, "--color", "auto"]).color)
        self.assertEqual(
            _parse_fit([*required, "--color", "1", "2", "3"]).color, (1, 2, 3)
        )
        with contextlib.redirect_stderr(io.StringIO()):
            for bad in (["1", "2"], ["1", "2", "256"], ["auto", "1"]):
                with self.assertRaises(SystemExit):
                    _parse_fit([*required, "--color", *bad])

    def test_cli_reports_input_errors_with_exit_code_2(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                main(
                    [
                        "--crops",
                        str(Path(directory) / "missing.json"),
                        "--manifest",
                        str(Path(directory) / "m.json"),
                        "--out",
                        str(Path(directory) / "out"),
                    ]
                )
            self.assertEqual(error.exception.code, 2)


class PhaseTests(unittest.TestCase):
    def test_phase_table_lists_row_and_line_phases_and_implied_scales(self):
        # Rows 1 and 12 are 231 native rows apart; origins 300.3 rows
        # apart imply s=1.3, and so do line centers 1 to 11.
        measured = {
            "candidates": [
                {
                    "name": "measured-bucket",
                    "evaluations": {
                        "in_sample": {
                            "fits": [
                                {"sample": "c1_slot1", "origin": [12.25, 3.5]},
                                {
                                    "sample": "c1_slot12",
                                    "origin": [12.5, 303.8],
                                },
                                {"sample": "c2_slot1", "origin": [2.0, 1.0]},
                            ]
                        }
                    },
                }
            ]
        }
        lines = [
            {
                "caseId": "c1",
                "scale": 1.3,
                "lines": [
                    {"line": 1, "center_row": 40.75},
                    {"line": 11, "center_row": 313.75},
                    {"line": 5, "center_row": math.nan},
                ],
            },
            {
                "caseId": "c3",
                "scale": 1.3,
                "lines": [{"line": 1, "center_row": 1.0}],
            },
        ]
        rows = phase_table(measured, "measured-bucket", lines)
        self.assertEqual([row["caseId"] for row in rows], ["c1"])
        self.assertEqual(rows[0]["rows"], {"1": [0.25, 0.5], "12": [0.5, 0.8]})
        self.assertEqual(rows[0]["lines"], {"1": 0.75, "11": 0.75})
        self.assertAlmostEqual(rows[0]["row_scale"], 1.3, places=4)
        self.assertAlmostEqual(rows[0]["line_scale"], 1.3, places=4)
        with self.assertRaises(ValueError):
            phase_table(measured, "bilinear", lines)
        with self.assertRaises(ValueError):
            phase_table(measured, "measured-bucket", lines, "holdout_set")
