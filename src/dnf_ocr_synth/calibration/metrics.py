"""Compare composites with captured crops the way dfragon binarizes.

The gray conversion and Otsu threshold are ported from dfragon's
``packages/client-core/src/lib/nickname-pixels.ts`` so mismatch counts
describe what the OCR model would see.
"""

import math
from dataclasses import dataclass

import numpy as np

GRAY_WEIGHTS = (9798, 19235, 3735)
GRAY_SHIFT = 15
MIN_CLASS_PROBABILITY = 2.0**-23
NICKNAME_BACKGROUND = 255


def bt601_gray(rgba: np.ndarray) -> np.ndarray:
    """Convert (..., 3+) uint8 pixels with the product's luma."""
    pixels = np.asarray(rgba, dtype=np.int64)
    red, green, blue = GRAY_WEIGHTS
    gray = (
        red * pixels[..., 0]
        + green * pixels[..., 1]
        + blue * pixels[..., 2]
        + (1 << (GRAY_SHIFT - 1))
    ) >> GRAY_SHIFT

    return gray.astype(np.uint8)


def otsu_threshold(histogram: list[int], count: int, total: int) -> int:
    """Port findOtsuThreshold, keeping its float operation order."""
    scale = 1 / count
    mean = total * scale
    dark_probability = 0.0
    dark_mean = 0.0
    max_variance = 0.0
    threshold = 0
    for gray, bin_count in enumerate(histogram):
        probability = bin_count * scale
        dark_mean *= dark_probability
        dark_probability += probability
        light_probability = 1 - dark_probability
        if (
            min(dark_probability, light_probability) < MIN_CLASS_PROBABILITY
            or max(dark_probability, light_probability)
            > 1 - MIN_CLASS_PROBABILITY
        ):
            continue
        dark_mean = (dark_mean + gray * probability) / dark_probability
        light_mean = (mean - dark_probability * dark_mean) / light_probability
        difference = dark_mean - light_mean
        variance = (
            dark_probability * light_probability * difference * difference
        )
        if variance > max_variance:
            max_variance = variance
            threshold = gray

    return threshold


def binarize_nickname_pixels(rgba: np.ndarray) -> np.ndarray:
    """Port binarizeNicknamePixels; ink becomes 0 and the rest 255."""
    source = np.asarray(rgba, dtype=np.uint8)
    flat = source.reshape(-1, 4)
    output = np.empty_like(flat)
    if flat.size == 0:
        return output.reshape(source.shape)
    gray = bt601_gray(flat)
    histogram = np.bincount(gray, minlength=256).tolist()
    threshold = otsu_threshold(histogram, gray.size, int(gray.sum()))
    value = np.where(gray > threshold, 0, NICKNAME_BACKGROUND)
    output[:, :3] = value[:, None]
    output[:, 3] = 255

    return output.reshape(source.shape)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Grow a boolean map by a square neighborhood."""
    padded = np.pad(mask, radius)
    result = np.zeros_like(mask)
    size = 2 * radius + 1
    height, width = mask.shape
    for dy in range(size):
        for dx in range(size):
            result |= padded[dy : dy + height, dx : dx + width]

    return result


def inpaint_rows(rgba: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Replace ``region`` pixels by interpolating along each row.

    Rows without any pixel outside the region are left as they are.
    """
    result = np.asarray(rgba, dtype=np.uint8).copy()
    columns = np.arange(result.shape[1])
    for row in range(result.shape[0]):
        known = ~region[row]
        if known.all() or not known.any():
            continue
        for channel in range(3):
            values = result[row, :, channel].astype(np.float64)
            result[row, ~known, channel] = np.rint(
                np.interp(columns[~known], columns[known], values[known])
            ).astype(np.uint8)

    return result


@dataclass(frozen=True)
class Reference:
    """Cache a captured crop with its gray, ink map, and background.

    The background is the crop with the text (the product's ink map
    grown by the outline and its blurred edge, about two texels)
    filled in from the same row. Compositing on it avoids blending a
    hypothesis over the real text's own anti-aliased edges.
    """

    rgba: np.ndarray
    gray: np.ndarray
    ink: np.ndarray
    background: np.ndarray

    @classmethod
    def from_rgba(cls, rgba: np.ndarray, scale: float) -> "Reference":
        """Prepare a (H, W, 4) uint8 crop captured at ``scale``."""
        pixels = np.asarray(rgba, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("crop must be an (H, W, 4) RGBA array.")
        ink = binarize_nickname_pixels(pixels)[..., 0] == 0
        text_region = dilate(ink, math.ceil(2 * scale) + 1)
        gray = bt601_gray(pixels)
        background = inpaint_rows(pixels, text_region)

        return cls(pixels, gray, ink, background)

    @property
    def ink_count(self) -> int:
        """Count pixels the product reads as ink."""
        return int(self.ink.sum())


@dataclass(frozen=True)
class Comparison:
    """Summarize one placement of a synthetic image over a crop."""

    mae: float
    mismatch: int
    ink: int

    @property
    def ratio(self) -> float:
        """Return mismatch pixels per crop ink pixel."""
        if self.ink:
            return self.mismatch / self.ink

        return math.inf


def composite_over(
    reference: Reference, synthetic: np.ndarray, origin: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Blend ``synthetic`` over the estimated background at ``origin``.

    Pixels the synthetic does not touch keep the captured crop.
    Returns the composite RGBA and a boolean map of crop pixels where
    the synthetic alpha was above zero.
    """
    crop = reference.rgba
    height, width = crop.shape[:2]
    source_height, source_width = synthetic.shape[:2]
    x0, y0 = origin
    left = max(x0, 0)
    top = max(y0, 0)
    right = min(x0 + source_width, width)
    bottom = min(y0 + source_height, height)
    composite = crop.copy()
    covered = np.zeros((height, width), dtype=bool)
    if right <= left or bottom <= top:
        return composite, covered
    patch = synthetic[top - y0 : bottom - y0, left - x0 : right - x0]
    touched = patch[..., 3] > 0
    alpha = patch[..., 3:4].astype(np.float64) / 255
    base = reference.background[top:bottom, left:right, :3].astype(np.float64)
    blended = base * (1 - alpha) + patch[..., :3].astype(np.float64) * alpha
    window = composite[top:bottom, left:right, :3]
    window[touched] = np.rint(blended).astype(np.uint8)[touched]
    covered[top:bottom, left:right] = touched

    return composite, covered


def compare(
    reference: Reference, synthetic: np.ndarray, origin: tuple[int, int]
) -> Comparison:
    """Measure gray MAE under the synthetic and Otsu mismatches."""
    composite, covered = composite_over(reference, synthetic, origin)
    region = dilate(covered)
    if region.any():
        difference = np.abs(
            reference.gray.astype(np.int16)
            - bt601_gray(composite).astype(np.int16)
        )
        mae = float(difference[region].mean())
    else:
        mae = math.inf
    binary = binarize_nickname_pixels(composite)[..., 0] == 0
    mismatch = int((binary != reference.ink).sum())

    return Comparison(mae, mismatch, reference.ink_count)


@dataclass(frozen=True)
class EdgeProfile:
    """Describe how a horizontal edge spreads over rows."""

    rows: list[int]
    values: list[float]
    low: float
    high: float
    rise_rows: float


def _crossing(values: list[float], level: float) -> float:
    for index in range(1, len(values)):
        before, after = values[index - 1], values[index]
        if before < level <= after:
            return index - 1 + (level - before) / (after - before)

    return math.nan


def edge_profile(
    image: np.ndarray,
    x: int,
    width: int,
    y: int,
    window: int = 3,
    channel: int = 0,
) -> EdgeProfile:
    """Average one channel across columns around a horizontal edge.

    ``rise_rows`` is the distance between the 10% and 90% crossings
    of the step. A crisp edge gives about 1 row; an edge scaled with
    the glyph filter spreads over about ``scale`` rows.
    """
    pixels = np.asarray(image)
    rows = list(range(y - window, y + window + 1))
    inside = [row for row in rows if 0 <= row < pixels.shape[0]]
    if len(inside) != len(rows):
        raise ValueError("The window reaches outside the image.")
    band = pixels[rows[0] : rows[-1] + 1, x : x + width, channel]
    if band.shape[1] != width:
        raise ValueError("The column range reaches outside the image.")
    values = band.astype(np.float64).mean(axis=1).tolist()
    low, high = min(values), max(values)
    if high - low < 1:
        return EdgeProfile(rows, values, low, high, math.nan)
    oriented = values if values[0] <= values[-1] else values[::-1]
    span = high - low
    rise = _crossing(oriented, low + 0.9 * span) - _crossing(
        oriented, low + 0.1 * span
    )

    return EdgeProfile(rows, values, low, high, rise)
