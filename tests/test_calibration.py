import contextlib
import io
import json
import math
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
    load_samples,
    main,
)
from dnf_ocr_synth.calibration.align import (
    best_candidate,
    estimate_text_origin,
    search_alignment,
)
from dnf_ocr_synth.calibration.metrics import (
    Reference,
    binarize_nickname_pixels,
    binarize_raid_nickname_pixels,
    binarize_raid_text_pixels,
    bt601_gray,
    compare,
    edge_profile,
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
from dnf_ocr_synth.render import _colorize, _compose

FIXTURES = Path(__file__).parent / "fixtures"
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
