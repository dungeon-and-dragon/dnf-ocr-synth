"""Estimate the game's resampling kernel from captured crops.

The enumerated pipelines fix the kernel shape (nearest, bilinear,
bicubic). This module instead models the scaled text as the native
outlined raster resampled with a separable, symmetric kernel given by
piecewise-linear taps in crop pixels, and solves the taps by
alternating least squares on the darkest channel of real crops. The
crop is modeled as ``background * (1 - A) + fill * C`` where ``A`` is
the resampled outline-plus-fill coverage and ``C`` the resampled fill
coverage, the same premultiplied composite the pipelines use.
"""

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from ..render import GlyphLayout, _compose
from .metrics import Reference

TAP_SPACING = 0.25
TAP_HALF_WIDTH = 4.0
NATIVE_PADDING = 2
COVERED_THRESHOLD = 0.02
REFINE_STEPS = (0.5, 0.25, 0.1, 0.05, 0.025)


@dataclass(frozen=True)
class KernelTaps:
    """Hold symmetric kernel samples at ``TAP_SPACING`` crop pixels."""

    x: tuple[float, ...]
    y: tuple[float, ...]

    @property
    def distances(self) -> list[float]:
        """Return the distance of each tap from the kernel center."""
        return [index * TAP_SPACING for index in range(len(self.x))]

    def integral(self, axis: str, scale: float) -> float:
        """Return the kernel area per source pixel.

        An area of 1 keeps the brightness of the native layer.
        """
        taps = self.x if axis == "x" else self.y

        return (taps[0] + 2 * sum(taps[1:])) * TAP_SPACING / scale


def tap_count() -> int:
    """Return how many taps span ``[0, TAP_HALF_WIDTH]``."""
    return int(round(TAP_HALF_WIDTH / TAP_SPACING)) + 1


def tent_taps(scale: float) -> KernelTaps:
    """Return the bilinear kernel as taps, the estimation's start."""
    taps = tuple(
        max(0.0, 1 - index * TAP_SPACING / scale)
        for index in range(tap_count())
    )

    return KernelTaps(taps, taps)


def native_layers(glyphs: GlyphLayout) -> tuple[np.ndarray, np.ndarray]:
    """Return the fill and the outline-plus-fill masks, padded by 2."""
    mask = _compose(glyphs.masks, glyphs.positions)
    ink = Image.new(
        "L",
        (mask.width + 2 * NATIVE_PADDING, mask.height + 2 * NATIVE_PADDING),
    )
    ink.paste(mask, (NATIVE_PADDING, NATIVE_PADDING))
    alpha = ink.filter(ImageFilter.MaxFilter(3))

    return np.asarray(ink) > 0, np.asarray(alpha) > 0


def tap_basis(distances: np.ndarray) -> np.ndarray:
    """Expand distances into piecewise-linear tap weights.

    The result has one more axis of ``tap_count()`` weights, so a
    kernel value is ``basis @ taps``; distances beyond the last tap
    get no weight.
    """
    position = np.abs(distances) / TAP_SPACING
    lower = np.floor(position).astype(int)
    fraction = position - lower
    count = tap_count()
    inside = lower < count - 1
    lower = np.where(inside, lower, 0)
    basis = np.zeros((*distances.shape, count))
    np.put_along_axis(
        basis,
        lower[..., None],
        np.where(inside, 1 - fraction, 0)[..., None],
        -1,
    )
    upper = np.zeros_like(basis)
    np.put_along_axis(
        upper,
        (lower + 1)[..., None],
        np.where(inside, fraction, 0)[..., None],
        -1,
    )

    return basis + upper


def sample_distances(
    output: int, native: int, scale: float, origin: float
) -> np.ndarray:
    """Return crop-pixel distances between output and native centers.

    Native pixel ``i`` is centered at ``origin + scale * (i + 0.5)``.
    """
    centers = np.arange(output)[:, None] + 0.5
    sources = origin + scale * (np.arange(native)[None, :] + 0.5)

    return centers - sources


def resample(
    fill: np.ndarray,
    alpha: np.ndarray,
    scale: float,
    origin: tuple[float, float],
    taps: KernelTaps,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage ``A`` and fill ``C`` of the native layers.

    ``size`` is ``(height, width)`` of the crop the layers land on.
    """
    height, width = size
    native_height, native_width = fill.shape
    weights_x = tap_basis(
        sample_distances(width, native_width, scale, origin[0])
    ) @ np.array(taps.x)
    weights_y = tap_basis(
        sample_distances(height, native_height, scale, origin[1])
    ) @ np.array(taps.y)
    coverage = weights_y @ alpha.astype(np.float64) @ weights_x.T
    filled = weights_y @ fill.astype(np.float64) @ weights_x.T

    return coverage, filled


def model_composite(
    reference: Reference,
    coverage: np.ndarray,
    filled: np.ndarray,
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Composite ``background * (1 - A) + color * C`` over the crop.

    Pixels the kernel does not reach keep the captured crop, so an
    overshooting kernel is kept rather than clipped to the color.
    Returns the RGBA composite and the map of reached pixels.
    """
    covered = coverage > 0
    background = reference.background[..., :3].astype(np.float64)
    blended = (
        background * (1 - coverage[..., None])
        + np.array(color, dtype=np.float64) * filled[..., None]
    )
    composite = reference.rgba.copy()
    composite[covered, :3] = np.rint(np.clip(blended, 0, 255)).astype(
        np.uint8
    )[covered]

    return composite, covered


def _darkest(rgba: np.ndarray) -> np.ndarray:
    return np.asarray(rgba)[..., :3].min(axis=2).astype(np.float64)


@dataclass(frozen=True)
class KernelSample:
    """Pair a crop with its native layers and estimated placement."""

    reference: Reference
    fill: np.ndarray
    alpha: np.ndarray
    fill_value: float
    scale: float
    origin: tuple[float, float]

    def model(self, taps: KernelTaps) -> tuple[np.ndarray, np.ndarray]:
        """Return the modeled darkest channel and the coverage map."""
        coverage, filled = resample(
            self.fill,
            self.alpha,
            self.scale,
            self.origin,
            taps,
            self.reference.ink.shape,
        )
        background = _darkest(self.reference.background)
        modeled = background * (1 - coverage) + self.fill_value * filled

        return np.clip(modeled, 0, 255), coverage


def _placement_error(
    reference: Reference,
    fill: np.ndarray,
    alpha: np.ndarray,
    fill_value: float,
    scale: float,
    origin: tuple[float, float],
    taps: KernelTaps,
) -> float:
    sample = KernelSample(reference, fill, alpha, fill_value, scale, origin)
    modeled, coverage = sample.model(taps)
    covered = coverage > COVERED_THRESHOLD
    if not covered.any():
        return math.inf
    difference = modeled - _darkest(reference.rgba)

    return float((difference[covered] ** 2).mean())


def align_sample(
    reference: Reference,
    glyphs: GlyphLayout,
    fill_value: float,
    scale: float,
    taps: KernelTaps | None = None,
) -> KernelSample:
    """Place the native layers on the crop with sub-pixel precision.

    Every integer origin keeping the layers inside the crop is scored
    by the squared darkest-channel error under the modeled coverage,
    then the best is refined by ``REFINE_STEPS``.
    """
    taps = tent_taps(scale) if taps is None else taps
    fill, alpha = native_layers(glyphs)
    height, width = reference.ink.shape
    span_x = width - fill.shape[1] * scale
    span_y = height - fill.shape[0] * scale
    if span_x < 0 or span_y < 0:
        raise ValueError("The native layers do not fit inside the crop.")

    def error(origin: tuple[float, float]) -> float:
        return _placement_error(
            reference, fill, alpha, fill_value, scale, origin, taps
        )

    best = min(
        (
            (error((x, y)), (float(x), float(y)))
            for y in range(math.floor(span_y) + 1)
            for x in range(math.floor(span_x) + 1)
        ),
        key=lambda item: item[0],
    )
    for step in REFINE_STEPS:
        while True:
            x, y = best[1]
            neighbors = [
                (x + dx, y + dy)
                for dy in (-step, 0.0, step)
                for dx in (-step, 0.0, step)
                if (dx, dy) != (0.0, 0.0)
            ]
            candidate = min(
                ((error(origin), origin) for origin in neighbors),
                key=lambda item: item[0],
            )
            if candidate[0] >= best[0]:
                break

            best = candidate

    return KernelSample(reference, fill, alpha, fill_value, scale, best[1])


def _solve_axis(
    samples: list[KernelSample], taps: KernelTaps, axis: str
) -> tuple[float, ...]:
    rows = []
    targets = []
    for sample in samples:
        height, width = sample.reference.ink.shape
        native_height, native_width = sample.fill.shape
        background = _darkest(sample.reference.background)
        target = _darkest(sample.reference.rgba) - background
        _, coverage = sample.model(taps)
        selected = coverage > COVERED_THRESHOLD
        fill = sample.fill.astype(np.float64) * sample.fill_value
        alpha = sample.alpha.astype(np.float64)
        if axis == "x":
            fixed = tap_basis(
                sample_distances(
                    height, native_height, sample.scale, sample.origin[1]
                )
            ) @ np.array(taps.y)
            basis = tap_basis(
                sample_distances(
                    width, native_width, sample.scale, sample.origin[0]
                )
            )
            filled = np.einsum("yi,xik->yxk", fixed @ fill, basis)
            covered = np.einsum("yi,xik->yxk", fixed @ alpha, basis)
        else:
            fixed = tap_basis(
                sample_distances(
                    width, native_width, sample.scale, sample.origin[0]
                )
            ) @ np.array(taps.x)
            basis = tap_basis(
                sample_distances(
                    height, native_height, sample.scale, sample.origin[1]
                )
            )
            filled = np.einsum("jx,yjk->yxk", fill @ fixed.T, basis)
            covered = np.einsum("jx,yjk->yxk", alpha @ fixed.T, basis)
        design = filled - background[..., None] * covered
        rows.append(design[selected])
        targets.append(target[selected])
    solution = np.linalg.lstsq(
        np.concatenate(rows), np.concatenate(targets), rcond=None
    )[0]

    return tuple(float(value) for value in solution)


def estimate_kernel(
    samples: list[KernelSample], scale: float, iterations: int = 3
) -> KernelTaps:
    """Fit the x and y taps alternately, starting from bilinear."""
    if not samples:
        raise ValueError("Kernel estimation needs at least one sample.")

    taps = tent_taps(scale)
    for _ in range(iterations):
        taps = KernelTaps(_solve_axis(samples, taps, "x"), taps.y)
        taps = KernelTaps(taps.x, _solve_axis(samples, taps, "y"))

    return taps
