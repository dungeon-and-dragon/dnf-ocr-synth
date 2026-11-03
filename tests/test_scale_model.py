import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dnf_ocr_synth import FontPaths, Renderer, ScaleModel, SeparableKernel
from dnf_ocr_synth.scale_model import (
    AFFINE_BILINEAR,
    CALIBRATED_V1,
    resample_layers,
    resolve_scale_model,
)

WINDOWS_FONTS = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
UTTUM = os.environ.get("DNF_SYNTH_UTTUM")
FONTS = FontPaths(
    gulim=WINDOWS_FONTS / "gulim.ttc",
    batang=WINDOWS_FONTS / "batang.ttc",
    uttum=Path(UTTUM) if UTTUM else None,
)
GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures/render-0.1.2.json").read_text(
        encoding="utf-8"
    )
)
# The frozen measured-native taps of dfragon #1144 (FROZEN.md table).
FROZEN_TAPS_X = (
    0.9900, 0.9862, 0.9671, 0.8604, 0.5248, 0.2136, 0.1457, 0.1401, 0.0337,
    -0.0556, -0.0451, -0.0169, 0.0039, 0.0059, 0.0120, 0.0113, 0.0010,
    0.0007, 0.0005, 0.0006, 0.0016,
)  # fmt: skip
FROZEN_TAPS_Y = (
    0.8931, 0.8923, 0.8750, 0.7768, 0.4728, 0.1962, 0.1328, 0.1275, 0.0294,
    -0.0506, -0.0404, -0.0190, 0.0036, 0.0030, 0.0047, 0.0053, 0.0004,
    -0.0009, -0.0000, -0.0013, 0.0048,
)  # fmt: skip
TENT = SeparableKernel(
    tuple(1 - index / 8 for index in range(9)),
    tuple(1 - index / 8 for index in range(9)),
    0.125,
)


def alpha_values(image):
    return set(image.getchannel("A").get_flattened_data())


class ScaleModelTests(unittest.TestCase):
    def test_names_resolve_and_a_model_passes_through(self):
        self.assertIs(resolve_scale_model("affine-bilinear"), AFFINE_BILINEAR)
        self.assertIs(resolve_scale_model("calibrated-v1"), CALIBRATED_V1)
        custom = ScaleModel("tent", TENT, "raid-name")
        self.assertIs(resolve_scale_model(custom), custom)
        with self.assertRaisesRegex(ValueError, "scale_model"):
            resolve_scale_model("bicubic")
        with self.assertRaisesRegex(ValueError, "glyph_layout"):
            ScaleModel("bad", glyph_layout="centered")

    def test_calibrated_v1_taps_match_the_frozen_table(self):
        kernel = CALIBRATED_V1.kernel
        self.assertEqual(kernel.spacing, 0.125)
        self.assertEqual(kernel.reach, 2.5)
        self.assertEqual(
            tuple(round(value, 4) for value in kernel.x), FROZEN_TAPS_X
        )
        self.assertEqual(
            tuple(round(value, 4) for value in kernel.y), FROZEN_TAPS_Y
        )
        self.assertEqual(CALIBRATED_V1.glyph_layout, "raid-name")

    def test_resampling_weighs_native_pixels_by_the_kernel(self):
        # One lit pixel in a 3x1 layer, scaled by 2: output pixel x is
        # (x + 0.5 - 3) / 2 native px from its center.
        (layer,) = resample_layers([[[0.0, 1.0, 0.0]]], 2, TENT, (6, 2))
        expected = [
            max(0.0, 1 - abs((x + 0.5 - 3) / 2)) * max(0.0, 1 - 0.25)
            for x in range(6)
        ]
        for row in layer:
            for value, want in zip(row, expected):
                self.assertAlmostEqual(value, want)

    def test_resampling_keeps_negative_lobes(self):
        lobe = SeparableKernel((1.0, -0.5, 0.0), (1.0, 0.0, 0.0), 0.5)
        (layer,) = resample_layers([[[0.0, 1.0, 0.0]]], 1, lobe, (3, 1))
        self.assertEqual(layer, [[0.0, 1.0, 0.0]])
        (layer,) = resample_layers([[[0.0, 1.0, 0.0]]], 2, lobe, (6, 2))
        self.assertLess(min(layer[0]), 0)


class RenderInputTests(unittest.TestCase):
    def test_bad_scale_models_fail_before_loading_fonts(self):
        renderer = Renderer(FontPaths())
        for options, message in (
            ({"scale_model": "missing"}, "scale_model must be"),
            ({"scale_model": "calibrated-v1", "scale": 0.9}, "at least 1"),
        ):
            with self.subTest(options=options):
                with self.assertRaisesRegex(ValueError, message):
                    renderer.render("검신", **options)


@unittest.skipUnless(
    FONTS.gulim.is_file() and FONTS.batang.is_file() and UTTUM,
    "Set DNF_SYNTH_UTTUM and supply the local Windows fonts",
)
class RenderScaleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.renderer = Renderer(FONTS)

    def test_default_model_keeps_the_0_1_2_pixels(self):
        for sample in GOLDEN["samples"]:
            for options in ({}, {"scale_model": "affine-bilinear"}):
                with self.subTest(sample=sample, options=options):
                    image = self.renderer.render(
                        sample["text"], scale=sample["scale"], **options
                    ).image
                    self.assertEqual(list(image.size), sample["size"])
                    self.assertEqual(
                        hashlib.sha256(image.tobytes()).hexdigest(),
                        sample["sha256"],
                    )

    def test_metadata_names_the_model_and_version(self):
        default = self.renderer.render("검신", scale=1.8).metadata
        self.assertEqual(default["schema_version"], 2)
        self.assertEqual(default["renderer_version"], "0.2.0")
        self.assertEqual(default["resampling"], "bilinear-premultiplied")
        self.assertEqual(default["padding_px"], 2)
        self.assertEqual(
            default["scale_model"],
            {
                "name": "affine-bilinear",
                "glyph_layout": "metrics",
                "kernel": None,
            },
        )
        calibrated = self.renderer.render(
            "검신", scale=1.8, scale_model="calibrated-v1"
        ).metadata
        self.assertEqual(
            calibrated["resampling"], "separable-kernel-premultiplied"
        )
        self.assertEqual(calibrated["padding_px"], 3)
        model = calibrated["scale_model"]
        self.assertEqual(model["name"], "calibrated-v1")
        self.assertEqual(model["glyph_layout"], "raid-name")
        self.assertEqual(
            model["kernel"]["taps_x"], list(CALIBRATED_V1.kernel.x)
        )
        self.assertEqual(
            self.renderer.render("검신", scale_model="calibrated-v1").metadata[
                "resampling"
            ],
            "native",
        )

    def test_calibrated_v1_spaces_hangul_by_ink_width_plus_one(self):
        text = "가나다"
        widths = [mask.width for mask in self.renderer.layout(text).masks]
        default = self.renderer.render(text).metadata["glyphs"]
        calibrated = self.renderer.render(
            text, scale_model="calibrated-v1"
        ).metadata["glyphs"]
        self.assertEqual(
            [glyph["advance_px"] for glyph in default], [11, 11, 11]
        )
        self.assertEqual(
            [glyph["advance_px"] for glyph in calibrated],
            [width + 1 for width in widths],
        )
        self.assertNotEqual(widths, [10, 10, 10])

    def test_calibrated_v1_at_scale_one_is_the_outlined_native_raster(self):
        sample = self.renderer.render(
            "가나", scale_model="calibrated-v1", color=(214, 197, 146)
        )
        image = sample.image
        width, height = sample.native_mask.size
        self.assertEqual(image.size, (width + 6, height + 6))
        self.assertEqual(alpha_values(image), {0, 255})
        self.assertEqual(
            set(image.get_flattened_data()) - {(0, 0, 0, 0)},
            {(214, 197, 146, 255), (0, 0, 0, 255)},
        )

    def test_calibrated_v1_size_and_alpha_range(self):
        for scale in (1.0733, 1.25, 1.8):
            with self.subTest(scale=scale):
                sample = self.renderer.render(
                    "검신조무래기", scale=scale, scale_model="calibrated-v1"
                )
                image = sample.image
                width, height = sample.native_mask.size
                self.assertEqual(
                    image.size,
                    (
                        math.ceil((width + 6) * scale),
                        math.ceil((height + 6) * scale),
                    ),
                )
                alpha = image.getchannel("A")
                self.assertEqual(alpha.getextrema(), (0, 255))
                self.assertTrue(alpha_values(image) - {0, 255})
                # The kernel reaches zero before the 3px padding ends.
                border = Image.new("L", image.size, 255)
                border.paste(0, (1, 1, image.width - 1, image.height - 1))
                self.assertEqual(
                    set(
                        value
                        for value, edge in zip(
                            alpha.get_flattened_data(),
                            border.get_flattened_data(),
                        )
                        if edge
                    ),
                    {0},
                )

    def test_cli_writes_a_calibrated_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "dnf_ocr_synth",
                    "검신",
                    "--gulim",
                    str(FONTS.gulim),
                    "--ui-percent",
                    "100",
                    "--scale-model",
                    "calibrated-v1",
                    "--output",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with Image.open(path) as image:
                metadata = json.loads(image.info["dnf_ocr_synth"])
        self.assertEqual(metadata["scale"], 1.8)
        self.assertEqual(metadata["scale_model"]["name"], "calibrated-v1")
