"""Compare composites with captured crops the way dfragon binarizes.

The gray conversions, Otsu threshold, and the raid frame-line and
detached-pixel removal are ported from dfragon's
``packages/client-core/src/lib/nickname-pixels.ts`` so mismatch counts
describe what the OCR model would see. ``SCREENS`` names the two crop
sources with their binarization.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

GRAY_WEIGHTS = (9798, 19235, 3735)
GRAY_SHIFT = 15
MIN_CLASS_PROBABILITY = 2.0**-23
NICKNAME_BACKGROUND = 255
FRAME_EDGE_MARGIN_RATIO = 1 / 5
FRAME_LINE_WIDTH_RATIO = 3 / 4
RAID_DETACHED_GAP_RATIO = 4 / 17


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


def darkest_channel(rgba: np.ndarray) -> np.ndarray:
    """Return the minimum of the RGB channels of (..., 3+) pixels."""
    pixels = np.asarray(rgba, dtype=np.uint8)

    return pixels[..., :3].min(axis=-1)


def _binarize_text_pixels(
    rgba: np.ndarray, brightness: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    source = np.asarray(rgba, dtype=np.uint8)
    flat = source.reshape(-1, 4)
    output = np.empty_like(flat)
    if flat.size == 0:
        return output.reshape(source.shape)
    gray = brightness(flat)
    histogram = np.bincount(gray, minlength=256).tolist()
    threshold = otsu_threshold(histogram, gray.size, int(gray.sum()))
    value = np.where(gray > threshold, 0, NICKNAME_BACKGROUND)
    output[:, :3] = value[:, None]
    output[:, 3] = 255

    return output.reshape(source.shape)


def binarize_nickname_pixels(rgba: np.ndarray) -> np.ndarray:
    """Port binarizeNicknamePixels; ink becomes 0 and the rest 255."""
    return _binarize_text_pixels(rgba, bt601_gray)


def binarize_raid_text_pixels(rgba: np.ndarray) -> np.ndarray:
    """Port binarizeRaidTextPixels: Otsu on the darkest channel."""
    return _binarize_text_pixels(rgba, darkest_channel)


def _frame_line_pixels(ink: np.ndarray) -> np.ndarray:
    """Port excludeNicknameFrameLines on a boolean (H, W) ink map.

    Runs of ink at least 3/4 of the width long count as frame lines,
    but only in the rows within the top or bottom fifth of the crop.
    """
    height, width = ink.shape
    margin = math.floor(height * FRAME_EDGE_MARGIN_RATIO)
    min_line_width = math.ceil(width * FRAME_LINE_WIDTH_RATIO)
    lines = np.zeros_like(ink)
    for y in (*range(margin), *range(height - margin, height)):
        start = 0
        for x in range(width + 1):
            if x < width and ink[y, x]:
                continue

            if x - start >= min_line_width:
                lines[y, start:x] = True
            start = x + 1

    return lines


def remove_raid_nickname_frame_lines(binary: np.ndarray) -> np.ndarray:
    """Port removeRaidNicknameFrameLines on a binarized crop."""
    result = np.asarray(binary, dtype=np.uint8).copy()
    lines = _frame_line_pixels(result[..., 0] < NICKNAME_BACKGROUND)
    result[lines, :3] = NICKNAME_BACKGROUND

    return result


@dataclass
class _ColumnCluster:
    left: int
    right: int
    count: int


def remove_detached_raid_nickname_pixels(binary: np.ndarray) -> np.ndarray:
    """Port removeDetachedRaidNicknamePixels on a binarized crop.

    Columns with ink are grouped into clusters separated by at least
    ``round(height * 4 / 17)`` empty columns, and every cluster but the
    one with the most ink pixels is cleared.
    """
    result = np.asarray(binary, dtype=np.uint8).copy()
    height, width = result.shape[:2]
    # JS Math.round rounds halves up, unlike Python's round().
    min_gap = max(1, math.floor(height * RAID_DETACHED_GAP_RATIO + 0.5))
    counts = (result[..., 0] < NICKNAME_BACKGROUND).sum(axis=0)
    clusters: list[_ColumnCluster] = []
    for x in range(width):
        count = int(counts[x])
        if count == 0:
            continue

        if clusters and x - clusters[-1].right - 1 < min_gap:
            clusters[-1].right = x
            clusters[-1].count += count
        else:
            clusters.append(_ColumnCluster(x, x, count))
    if not clusters:
        return result
    main = max(clusters, key=lambda cluster: cluster.count)
    for cluster in clusters:
        if cluster is not main:
            result[:, cluster.left : cluster.right + 1, :3] = (
                NICKNAME_BACKGROUND
            )

    return result


def binarize_raid_nickname_pixels(rgba: np.ndarray) -> np.ndarray:
    """Port binarizeRaidNicknamePixels on an (H, W, 4) crop."""
    source = np.asarray(rgba, dtype=np.uint8)
    if source.ndim != 3 or source.shape[2] != 4:
        raise ValueError("raid crop must be an (H, W, 4) RGBA array.")

    return remove_detached_raid_nickname_pixels(
        remove_raid_nickname_frame_lines(binarize_raid_text_pixels(source))
    )


SCREEN_BINARIZERS = {
    "party": binarize_nickname_pixels,
    "raid": binarize_raid_nickname_pixels,
}
SCREENS = tuple(SCREEN_BINARIZERS)


def binarize_for_screen(screen: str, rgba: np.ndarray) -> np.ndarray:
    """Binarize an (H, W, 4) crop as the product does on ``screen``."""
    if screen not in SCREEN_BINARIZERS:
        raise ValueError(f"screen must be one of {SCREENS}.")

    return SCREEN_BINARIZERS[screen](rgba)


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
    screen: str

    @classmethod
    def from_rgba(
        cls, rgba: np.ndarray, scale: float, screen: str
    ) -> "Reference":
        """Prepare a (H, W, 4) crop of ``screen`` captured at ``scale``.

        ``screen`` picks the product binarization: ``party`` uses the
        BT.601 Otsu and ``raid`` the darkest-channel Otsu with frame
        lines and detached pixels removed.
        """
        pixels = np.asarray(rgba, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("crop must be an (H, W, 4) RGBA array.")
        ink = binarize_for_screen(screen, pixels)[..., 0] == 0
        text_region = dilate(ink, math.ceil(2 * scale) + 1)
        gray = bt601_gray(pixels)
        background = inpaint_rows(pixels, text_region)

        return cls(pixels, gray, ink, background, screen)

    def binarize(self, rgba: np.ndarray) -> np.ndarray:
        """Binarize another crop-sized image with this crop's screen."""
        return binarize_for_screen(self.screen, rgba)

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

    return score_composite(reference, composite, covered)


def score_composite(
    reference: Reference, composite: np.ndarray, covered: np.ndarray
) -> Comparison:
    """Score a crop-sized composite; MAE covers ``covered`` plus 1px."""
    region = dilate(covered)
    if region.any():
        difference = np.abs(
            reference.gray.astype(np.int16)
            - bt601_gray(composite).astype(np.int16)
        )
        mae = float(difference[region].mean())
    else:
        mae = math.inf
    binary = reference.binarize(composite)[..., 0] == 0
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
    rows, values = _row_means(image, x, width, y, window, channel)
    low, high = min(values), max(values)
    if high - low < 1:
        return EdgeProfile(rows, values, low, high, math.nan)
    oriented = values if values[0] <= values[-1] else values[::-1]
    span = high - low
    rise = _crossing(oriented, low + 0.9 * span) - _crossing(
        oriented, low + 0.1 * span
    )

    return EdgeProfile(rows, values, low, high, rise)


def _row_means(
    image: np.ndarray, x: int, width: int, y: int, window: int, channel: int
) -> tuple[list[int], list[float]]:
    pixels = np.asarray(image)
    rows = list(range(y - window, y + window + 1))
    inside = [row for row in rows if 0 <= row < pixels.shape[0]]
    if len(inside) != len(rows):
        raise ValueError("The window reaches outside the image.")
    band = pixels[rows[0] : rows[-1] + 1, x : x + width, channel]
    if band.shape[1] != width:
        raise ValueError("The column range reaches outside the image.")

    return rows, band.astype(np.float64).mean(axis=1).tolist()


@dataclass(frozen=True)
class LineProfile:
    """Describe how a dark horizontal line spreads over rows."""

    rows: list[int]
    values: list[float]
    width_rows: float


def line_width(values: list[float]) -> float:
    """Measure the darkest dip's full width at half depth, in rows.

    Each side's half level sits between the dip and that side's
    brightest value, so a line between differently lit rows is still
    measured from its own edges. A native line of ``n`` rows scaled
    with the layer by ``s`` measures about ``n * s``, while a line
    redrawn crisp measures a whole number of rows.
    """
    center = min(range(len(values)), key=values.__getitem__)
    low = values[center]
    above = values[: center + 1][::-1]
    below = values[center:]
    if len(above) < 2 or len(below) < 2:
        return math.nan
    up = _crossing(above, (low + max(above)) / 2)
    down = _crossing(below, (low + max(below)) / 2)

    return up + down


def line_center(values: list[float]) -> float:
    """Locate the darkest dip's center, in rows from the first value.

    The center is the midpoint between the two half-depth crossings
    ``line_width`` measures, so it carries the line's sub-pixel phase.
    """
    center = min(range(len(values)), key=values.__getitem__)
    low = values[center]
    above = values[: center + 1][::-1]
    below = values[center:]
    if len(above) < 2 or len(below) < 2:
        return math.nan
    up = _crossing(above, (low + max(above)) / 2)
    down = _crossing(below, (low + max(below)) / 2)

    return center + (down - up) / 2


def line_profile(
    image: np.ndarray,
    x: int,
    width: int,
    y: int,
    window: int = 3,
    channel: int = 0,
) -> LineProfile:
    """Average one channel across columns around a dark line."""
    rows, values = _row_means(image, x, width, y, window, channel)

    return LineProfile(rows, values, line_width(values))
