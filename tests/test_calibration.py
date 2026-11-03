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

from dnf_ocr_synth.calibration.__main__ import load_samples, main
from dnf_ocr_synth.calibration.align import (
    best_candidate,
    estimate_text_origin,
    search_alignment,
)
from dnf_ocr_synth.calibration.metrics import (
    Reference,
    binarize_nickname_pixels,
    bt601_gray,
    compare,
    edge_profile,
    otsu_threshold,
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
        # Vectors were produced by dfragon's binarizeNicknamePixels from
        # synthetic arrays; they hold no screenshot pixels.
        fixture = json.loads(
            (FIXTURES / "otsu-vectors.json").read_text(encoding="utf-8")
        )
        self.assertGreaterEqual(len(fixture["vectors"]), 5)
        for vector in fixture["vectors"]:
            with self.subTest(name=vector["name"]):
                rgba = np.array(vector["rgba"], dtype=np.uint8).reshape(
                    vector["height"], vector["width"], 4
                )
                output = binarize_nickname_pixels(rgba)
                self.assertEqual(output.reshape(-1).tolist(), vector["output"])

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
                reference = Reference.from_rgba(crop, 1.8)
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
            Reference.from_rgba(crop, 1.8),
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
        reference = Reference.from_rgba(crop, 1.8)
        exact = compare(reference, self.synthetic, (5, 5))
        shifted = compare(reference, self.synthetic, (6, 5))
        self.assertEqual(exact.mismatch, 0)
        self.assertGreater(shifted.mismatch, 0)
        self.assertGreater(shifted.mae, exact.mae)

    def test_placement_outside_the_crop_has_infinite_mae(self):
        reference = Reference.from_rgba(self.background, 1.8)
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
            SampleInfo("a", "a", 1, "a.png", 1.0, "가", 0, "prtsc"),
            SampleInfo("b", "b", 1, "b.png", 1.82, "나", 100, "prtsc"),
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
            Reference.from_rgba(crop, 1.0), np.asarray(rendered.image), (3, 2)
        )
        self.assertEqual(sheet.size, (160, 242))
        samples = [SampleInfo("a", "a", 1, "a.png", 1.0, "가", 0, "prtsc")]
        fits = [Fit("a", DEFAULT_PIPELINE.name, (0, 0), (3, 2), 0.0, 0, 10)]
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "out"
            write_report(
                out,
                samples,
                fits,
                [],
                rank_pipelines(samples, fits, 5),
                {"pipeline_count": 1, "radius": 2},
            )
            payload = json.loads((out / "report.json").read_text("utf-8"))
            markdown = (out / "report.md").read_text("utf-8")
        self.assertEqual(payload["fits"][0]["ratio"], 0.0)
        self.assertEqual(payload["rankings"]["1.0"][0]["samples"], 1)
        self.assertIn(DEFAULT_PIPELINE.name, markdown)
        self.assertIn("| a | 가 | 1.00 |", markdown)


class SampleLoadingTests(unittest.TestCase):
    def write_inputs(self, directory):
        crops = Path(directory) / "crops"
        crops.mkdir()
        index = {
            "crops": [
                {
                    "caseId": "c1",
                    "slot": 1,
                    "file": "c1_slot1.png",
                    "scale": 1.29,
                    "uiPercent": None,
                    "capture": "ingame-jpg",
                },
                {
                    "caseId": "c1",
                    "slot": 2,
                    "file": "c1_slot2.png",
                    "scale": 1.29,
                    "uiPercent": None,
                    "capture": "ingame-jpg",
                },
                {
                    "caseId": "c1",
                    "slot": 3,
                    "file": "c1_slot3.png",
                    "scale": 1.29,
                    "uiPercent": None,
                    "capture": "ingame-jpg",
                },
            ]
        }
        (crops / "index.json").write_text(json.dumps(index), "utf-8")
        manifest = {
            "cases": [
                {
                    "id": "c1",
                    "slots": [
                        {"slot": 1, "text": "가나"},
                        {"slot": 2, "text": "다라", "uncertain": True},
                    ],
                }
            ]
        }
        path = Path(directory) / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")

        return crops / "index.json", path

    def test_joins_labels_and_skips_uncertain_or_unlabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(directory)
            samples, skipped = load_samples(index, manifest, False)
            self.assertEqual(
                [(s.id, s.text, s.bucket) for s in samples],
                [("c1_slot1", "가나", "1.2")],
            )
            self.assertEqual(
                [entry.sample for entry in skipped],
                ["c1_slot2", "c1_slot3"],
            )
            included, _ = load_samples(index, manifest, True)
            self.assertEqual(
                [s.id for s in included], ["c1_slot1", "c1_slot2"]
            )

    def test_missing_keys_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            index, manifest = self.write_inputs(directory)
            index.write_text('{"crops": [{"slot": 1}]}', "utf-8")
            with self.assertRaisesRegex(ValueError, "caseId"):
                load_samples(index, manifest, False)

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
