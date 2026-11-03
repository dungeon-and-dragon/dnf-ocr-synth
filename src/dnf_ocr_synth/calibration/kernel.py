"""Resample native text with measured or standard kernels.

The enumerated pipelines rasterize with Pillow. This module instead
models the scaled text as the native outlined raster resampled with a
separable, symmetric kernel, so every kernel shape meets the crops
under the same background, composite, alignment objective and
sub-pixel origin search. ``KernelTaps`` holds a measured kernel as
piecewise-linear taps, solved by alternating least squares on the
darkest channel of real crops; ``StandardKernel`` holds a fixed filter
shape such as bilinear. The crop is modeled as
``background * (1 - A) + fill * C`` where ``A`` is the resampled
outline-plus-fill coverage and ``C`` the resampled fill coverage, the
same premultiplied composite the pipelines use.
"""

import math
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from PIL import Image, ImageFilter

from ..render import GlyphLayout, _compose
from .metrics import Reference

# Taps sit every quarter crop pixel out to 4 crop px, or every eighth
# native pixel out to 2.5 native px, where the kernel reaches zero.
TAP_UNITS = ("crop", "native")
TAP_SPACING = {"crop": 0.25, "native": 0.125}
TAP_HALF_WIDTH = {"crop": 4.0, "native": 2.5}
NATIVE_PADDING = 2
COVERED_THRESHOLD = 0.02
REFINE_STEPS = (0.5, 0.25, 0.1, 0.05, 0.025)


class Kernel(Protocol):
    """Weight native pixels by crop-pixel distance to an output."""

    def weights(
        self, distances: np.ndarray, scale: float, axis: str
    ) -> np.ndarray:
        """Return the kernel value at each crop-pixel distance."""

    def integral(self, axis: str, scale: float) -> float:
        """Return the kernel area per source pixel."""


def tap_count(unit: str) -> int:
    """Return how many taps span ``[0, TAP_HALF_WIDTH[unit]]``."""
    return int(round(TAP_HALF_WIDTH[unit] / TAP_SPACING[unit])) + 1


@dataclass(frozen=True)
class KernelTaps:
    """Hold symmetric kernel samples at ``TAP_SPACING[unit]`` spacing.

    Distances are crop pixels for ``unit="crop"`` and source pixels for
    ``unit="native"``, so a native kernel is ``h(d / s)`` shared by
    every scale. Taps may be negative and need not integrate to 1.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    unit: str = "crop"

    def __post_init__(self) -> None:
        """Reject an unknown unit."""
        if self.unit not in TAP_UNITS:
            raise ValueError(f"unit must be one of {TAP_UNITS}.")

    @property
    def spacing(self) -> float:
        """Return the distance between taps in the kernel's unit."""
        return TAP_SPACING[self.unit]

    @property
    def distances(self) -> list[float]:
        """Return the distance of each tap from the kernel center."""
        return [index * self.spacing for index in range(len(self.x))]

    def axis(self, axis: str) -> tuple[float, ...]:
        """Return the taps of ``axis`` (``x`` or ``y``)."""
        if axis == "x":
            return self.x

        return self.y

    def unit_scale(self, scale: float) -> float:
        """Return how many crop pixels one tap unit spans."""
        if self.unit == "native":
            return scale

        return 1.0

    def weights(
        self, distances: np.ndarray, scale: float, axis: str
    ) -> np.ndarray:
        """Interpolate the taps at crop-pixel ``distances``."""
        taps = self.axis(axis)

        return tap_basis(
            distances / self.unit_scale(scale), self.spacing, len(taps)
        ) @ np.array(taps)

    def integral(self, axis: str, scale: float) -> float:
        """Return the kernel area per source pixel.

        The kernel is piecewise linear between taps and zero beyond the
        last one, so the trapezoid rule counts the last tap once.
        """
        taps = self.axis(axis)
        area = (taps[0] + 2 * sum(taps[1:-1]) + taps[-1]) * self.spacing

        return area * self.unit_scale(scale) / scale

    def unit_gain(self, scale: float) -> "KernelTaps":
        """Scale the x taps so the x and y areas multiply to 1."""
        product = self.integral("x", scale) * self.integral("y", scale)

        return replace(self, x=tuple(value / product for value in self.x))


def tent_taps(scale: float, unit: str = "crop") -> KernelTaps:
    """Return the bilinear kernel as taps, the estimation's start."""
    spacing = TAP_SPACING[unit]
    source = scale if unit == "crop" else 1.0
    taps = tuple(
        max(0.0, 1 - index * spacing / source)
        for index in range(tap_count(unit))
    )

    return KernelTaps(taps, taps, unit)


def _catmull_rom(distance: np.ndarray) -> np.ndarray:
    near = 1.5 * distance**3 - 2.5 * distance**2 + 1
    far = -0.5 * distance**3 + 2.5 * distance**2 - 4 * distance + 2

    return np.where(distance < 1, near, np.where(distance < 2, far, 0.0))


def _tent_area(upper: np.ndarray) -> np.ndarray:
    # Area of the unit tent from -1 to ``upper``.
    clipped = np.clip(upper, -1, 1)

    return np.where(
        clipped < 0,
        (clipped + 1) ** 2 / 2,
        0.5 + clipped - clipped**2 / 2,
    )


def _box_then_bilinear(distance: np.ndarray) -> np.ndarray:
    # A unit box prefilter followed by bilinear sampling is the tent
    # averaged over one source pixel.
    return _tent_area(distance + 0.5) - _tent_area(distance - 0.5)


# Kernel shapes in source pixels, from the resampling they describe.
# ``nearest2x-bilinear`` doubles the raster with nearest first, then
# samples it bilinearly, which flattens the tent to 0.25 px and ends it
# at 0.75 px.
STANDARD_SHAPES = {
    "nearest": lambda distance: (distance < 0.5).astype(np.float64),
    "bilinear": lambda distance: np.maximum(0.0, 1 - distance),
    "catmull-rom": _catmull_rom,
    "nearest2x-bilinear": lambda distance: np.clip(1.5 - 2 * distance, 0, 1),
    "box-bilinear": _box_then_bilinear,
}


@dataclass(frozen=True)
class StandardKernel:
    """Hold a fixed non-negative filter shape with an optional gain.

    ``gain`` multiplies the x weights, so the x and y areas multiply to
    ``gain``; 1 keeps the native brightness.
    """

    shape: str
    gain: float = 1.0

    def __post_init__(self) -> None:
        """Reject a shape outside ``STANDARD_SHAPES``."""
        if self.shape not in STANDARD_SHAPES:
            raise ValueError(f"shape must be one of {tuple(STANDARD_SHAPES)}.")

    def weights(
        self, distances: np.ndarray, scale: float, axis: str
    ) -> np.ndarray:
        """Evaluate the shape at crop-pixel ``distances``."""
        values = STANDARD_SHAPES[self.shape](np.abs(distances) / scale)
        if axis == "x":
            return values * self.gain

        return values

    def integral(self, axis: str, scale: float) -> float:
        """Integrate the shape numerically over source pixels."""
        distance = np.linspace(0, 3, 30001)
        area = 2 * np.trapezoid(
            STANDARD_SHAPES[self.shape](distance), distance
        )
        if axis == "x":
            return float(area * self.gain)

        return float(area)


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


def tap_basis(distances: np.ndarray, spacing: float, count: int) -> np.ndarray:
    """Expand distances into piecewise-linear tap weights.

    The result has one more axis of ``count`` weights, so a kernel
    value is ``basis @ taps``; distances beyond the last tap get no
    weight.
    """
    position = np.abs(distances) / spacing
    lower = np.floor(position).astype(int)
    fraction = position - lower
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
    kernel: Kernel,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage ``A`` and fill ``C`` of the native layers.

    ``size`` is ``(height, width)`` of the crop the layers land on.
    """
    height, width = size
    native_height, native_width = fill.shape
    weights_x = kernel.weights(
        sample_distances(width, native_width, scale, origin[0]), scale, "x"
    )
    weights_y = kernel.weights(
        sample_distances(height, native_height, scale, origin[1]), scale, "y"
    )
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

    def model(self, kernel: Kernel) -> tuple[np.ndarray, np.ndarray]:
        """Return the modeled darkest channel and the coverage map."""
        coverage, filled = resample(
            self.fill,
            self.alpha,
            self.scale,
            self.origin,
            kernel,
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
    kernel: Kernel,
) -> float:
    sample = KernelSample(reference, fill, alpha, fill_value, scale, origin)
    modeled, coverage = sample.model(kernel)
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
    kernel: Kernel | None = None,
) -> KernelSample:
    """Place the native layers on the crop with sub-pixel precision.

    Every integer origin keeping the layers inside the crop is scored
    by the squared darkest-channel error under the modeled coverage,
    then the best is refined by ``REFINE_STEPS``.
    """
    kernel = tent_taps(scale) if kernel is None else kernel
    fill, alpha = native_layers(glyphs)
    height, width = reference.ink.shape
    span_x = width - fill.shape[1] * scale
    span_y = height - fill.shape[0] * scale
    if span_x < 0 or span_y < 0:
        raise ValueError("The native layers do not fit inside the crop.")

    def error(origin: tuple[float, float]) -> float:
        return _placement_error(
            reference, fill, alpha, fill_value, scale, origin, kernel
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


def _axis_design(
    sample: KernelSample, taps: KernelTaps, axis: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the design rows, target and selection of one axis.

    With the other axis fixed, the modeled darkest channel is linear in
    this axis's taps: ``design @ taps`` approximates
    ``darkest - background`` on the pixels the current kernel covers.
    """
    height, width = sample.reference.ink.shape
    native_height, native_width = sample.fill.shape
    background = _darkest(sample.reference.background)
    target = _darkest(sample.reference.rgba) - background
    _, coverage = sample.model(taps)
    selected = coverage > COVERED_THRESHOLD
    fill = sample.fill.astype(np.float64) * sample.fill_value
    alpha = sample.alpha.astype(np.float64)
    unit = taps.unit_scale(sample.scale)
    count = len(taps.axis(axis))
    distances_x = sample_distances(
        width, native_width, sample.scale, sample.origin[0]
    )
    distances_y = sample_distances(
        height, native_height, sample.scale, sample.origin[1]
    )
    if axis == "x":
        fixed = taps.weights(distances_y, sample.scale, "y")
        basis = tap_basis(distances_x / unit, taps.spacing, count)
        filled = np.einsum("yi,xik->yxk", fixed @ fill, basis)
        covered = np.einsum("yi,xik->yxk", fixed @ alpha, basis)
    else:
        fixed = taps.weights(distances_x, sample.scale, "x")
        basis = tap_basis(distances_y / unit, taps.spacing, count)
        filled = np.einsum("jx,yjk->yxk", fill @ fixed.T, basis)
        covered = np.einsum("jx,yjk->yxk", alpha @ fixed.T, basis)
    design = filled - background[..., None] * covered

    return design[selected], target[selected], selected


def _solve_axis(
    samples: list[KernelSample], taps: KernelTaps, axis: str
) -> tuple[float, ...]:
    rows = []
    targets = []
    for sample in samples:
        design, target, _ = _axis_design(sample, taps, axis)
        rows.append(design)
        targets.append(target)
    solution = np.linalg.lstsq(
        np.concatenate(rows), np.concatenate(targets), rcond=None
    )[0]

    return tuple(float(value) for value in solution)


def estimate_kernel(
    samples: list[KernelSample],
    scale: float,
    iterations: int = 3,
    unit: str = "crop",
) -> KernelTaps:
    """Fit the x and y taps alternately, starting from bilinear.

    Nothing constrains the taps: they may go negative and the areas
    may differ from 1, so the fit absorbs any brightness gain.
    """
    if not samples:
        raise ValueError("Kernel estimation needs at least one sample.")

    taps = tent_taps(scale, unit)
    for _ in range(iterations):
        taps = replace(taps, x=_solve_axis(samples, taps, "x"))
        taps = replace(taps, y=_solve_axis(samples, taps, "y"))

    return taps


def fit_gain(samples: list[KernelSample], kernel: StandardKernel) -> float:
    """Return the least-squares gain of a standard kernel at gain 1.

    The modeled darkest channel is linear in the gain, so one scalar
    fits ``darkest - background`` over the covered pixels of every
    sample.
    """
    if not samples:
        raise ValueError("Gain estimation needs at least one sample.")

    unit = replace(kernel, gain=1.0)
    numerator = 0.0
    denominator = 0.0
    for sample in samples:
        coverage, filled = resample(
            sample.fill,
            sample.alpha,
            sample.scale,
            sample.origin,
            unit,
            sample.reference.ink.shape,
        )
        background = _darkest(sample.reference.background)
        selected = coverage > COVERED_THRESHOLD
        design = (sample.fill_value * filled - background * coverage)[selected]
        target = (_darkest(sample.reference.rgba) - background)[selected]
        numerator += float(design @ target)
        denominator += float(design @ design)
    if denominator == 0:
        raise ValueError("The kernel covers no pixel of the samples.")

    return numerator / denominator
