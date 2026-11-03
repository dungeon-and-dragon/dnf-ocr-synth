import contextlib
import io
import unittest

from dnf_ocr_synth.cli import parse_args


class ColorArgumentTests(unittest.TestCase):
    def test_default_and_custom_rgb(self):
        arguments = ["검신", "--output", "sample.png"]
        self.assertEqual(parse_args(arguments).color, (75, 209, 255))
        self.assertEqual(
            parse_args(arguments + ["--color", "255", "128", "0"]).color,
            (255, 128, 0),
        )

    def test_invalid_rgb_exits_before_rendering(self):
        for channels in (
            ["-1", "0", "0"],
            ["256", "0", "0"],
            ["1.5", "0", "0"],
            ["255", "0"],
            ["255", "0", "0", "255"],
        ):
            with self.subTest(channels=channels):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_args(
                            ["검신", "--output", "sample.png", "--color"]
                            + channels
                        )
                self.assertEqual(error.exception.code, 2)


class ScaleModelArgumentTests(unittest.TestCase):
    def test_default_and_calibrated_model(self):
        arguments = ["검신", "--output", "sample.png"]
        self.assertEqual(parse_args(arguments).scale_model, "affine-bilinear")
        self.assertEqual(
            parse_args(
                arguments + ["--scale-model", "calibrated-v1"]
            ).scale_model,
            "calibrated-v1",
        )
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args(arguments + ["--scale-model", "bicubic"])
        self.assertEqual(error.exception.code, 2)
