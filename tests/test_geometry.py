import math
import unittest

from dnf_ocr_synth import CropGeometry, name_crop_geometry, name_text_left
from dnf_ocr_synth.geometry import round_pixel


class NameCropGeometryTests(unittest.TestCase):
    def test_scale_one_cuts_86_by_17_cells_21_rows_apart(self):
        self.assertEqual(name_crop_geometry(1), CropGeometry(182, 21, 86, 17))
        self.assertEqual(
            name_crop_geometry(1, 12, (100, 50)),
            CropGeometry(282, 50 + 21 + 21 * 11, 86, 17),
        )

    def test_scaled_rows_match_a_captured_crop_index(self):
        # Boxes dfragon cut from a 1199x674 UI 100% capture (s=1.125)
        # whose raid heading sat at (567, 112).
        for row, box in (
            (1, CropGeometry(772, 136, 96, 19)),
            (4, CropGeometry(772, 206, 96, 20)),
            (12, CropGeometry(772, 396, 96, 19)),
        ):
            with self.subTest(row=row):
                self.assertEqual(
                    name_crop_geometry(1.125, row, (567, 112)), box
                )

    def test_exact_halves_round_to_the_even_pixel(self):
        self.assertEqual(
            [round_pixel(value) for value in (2.5, 3.5, -0.5, 2.6, 2.4)],
            [2, 4, 0, 3, 2],
        )
        # A few ulps above a half still counts as the half.
        self.assertEqual(round_pixel(math.nextafter(2.5, 3)), 2)
        self.assertEqual(
            round_pixel(math.nextafter(math.nextafter(2.5, 3), 3)), 2
        )
        self.assertEqual(round_pixel(2.5 + 1e-9), 3)

    def test_rows_outside_the_window_and_bad_scales_are_rejected(self):
        for row, scale in ((0, 1), (13, 1), (1, 0), (1, float("nan"))):
            with self.subTest(row=row, scale=scale):
                with self.assertRaises(ValueError):
                    name_crop_geometry(scale, row)


class NameTextLeftTests(unittest.TestCase):
    def test_text_is_centered_and_an_odd_half_rounds_as_told(self):
        self.assertEqual(name_text_left(84), 1)
        self.assertEqual(name_text_left(83, "up"), 2)
        self.assertEqual(name_text_left(83, "down"), 1)
        with self.assertRaises(ValueError):
            name_text_left(83, "nearest")
