"""Choose how a native nickname raster is scaled by the UI scale.

``affine-bilinear`` is the 0.1.2 behavior: a premultiplied Pillow
bilinear affine transform of the outlined raster. ``calibrated-v1``
resamples the outlined raster with the separable kernel measured on
captured raid-window name cells (dfragon #1144) and lays the name out
the way that window does. Its taps are the frozen fit, shipped as
package data and never refitted here.
"""

import json
import math
from dataclasses import dataclass
from importlib import resources

GLYPH_LAYOUTS = ("metrics", "raid-name")


@dataclass(frozen=True)
class SeparableKernel:
    """Hold symmetric kernel taps in native pixels.

    Tap ``k`` is the weight at ``k * spacing`` native px from a source
    pixel center; values between taps are linear and the kernel is
    zero from the last tap on. The same taps serve every scale, so an
    output pixel ``d`` crop px away weighs ``h(d / scale)``. Taps may
    be negative and need not integrate to 1.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    spacing: float

    def __post_init__(self) -> None:
        """Reject taps that cannot describe a kernel."""
        if len(self.x) < 2 or len(self.x) != len(self.y):
            raise ValueError("kernel needs two or more taps per axis.")
        if not self.spacing > 0:
            raise ValueError("kernel tap spacing must be positive.")

    @property
    def reach(self) -> float:
        """Return the native distance where the kernel reaches zero."""
        return (len(self.x) - 1) * self.spacing


@dataclass(frozen=True)
class ScaleModel:
    """Name a scaling method and the glyph layout it is paired with.

    Without a ``kernel`` the outlined raster is scaled by the 0.1.2
    bilinear affine transform. ``glyph_layout`` is ``metrics`` (the
    renderer's font layout) or ``raid-name`` (the raid window's name
    cell spacing).
    """

    name: str
    kernel: SeparableKernel | None = None
    glyph_layout: str = "metrics"

    def __post_init__(self) -> None:
        """Reject an unknown glyph layout."""
        if self.glyph_layout not in GLYPH_LAYOUTS:
            raise ValueError(f"glyph_layout must be one of {GLYPH_LAYOUTS}.")

    def metadata(self) -> dict:
        """Describe the model as JSON-friendly values."""
        kernel = None
        if self.kernel is not None:
            kernel = {
                "unit": "native",
                "tap_spacing_px": self.kernel.spacing,
                "taps_x": list(self.kernel.x),
                "taps_y": list(self.kernel.y),
            }

        return {
            "name": self.name,
            "glyph_layout": self.glyph_layout,
            "kernel": kernel,
        }


AFFINE_BILINEAR = ScaleModel("affine-bilinear")


def _load_calibrated_v1() -> ScaleModel:
    data = json.loads(
        resources.files(__package__)
        .joinpath("data/calibrated-v1.json")
        .read_text(encoding="utf-8")
    )
    kernel = SeparableKernel(
        tuple(data["taps_x"]), tuple(data["taps_y"]), data["tap_spacing"]
    )

    return ScaleModel(data["name"], kernel, "raid-name")


CALIBRATED_V1 = _load_calibrated_v1()
SCALE_MODELS = {
    model.name: model for model in (AFFINE_BILINEAR, CALIBRATED_V1)
}


def resolve_scale_model(value: "str | ScaleModel") -> ScaleModel:
    """Return the named model, or ``value`` when it is a model."""
    if isinstance(value, ScaleModel):
        return value

    if value not in SCALE_MODELS:
        raise ValueError(f"scale_model must be one of {tuple(SCALE_MODELS)}.")

    return SCALE_MODELS[value]


def _tap_value(taps: tuple[float, ...], spacing: float, distance: float):
    position = abs(distance) / spacing
    lower = math.floor(position)
    if lower >= len(taps) - 1:
        return 0.0

    fraction = position - lower

    return taps[lower] * (1 - fraction) + taps[lower + 1] * fraction


def _axis_weights(
    taps: tuple[float, ...],
    spacing: float,
    native: int,
    output: int,
    scale: float,
) -> list[list[tuple[int, float]]]:
    """List the nonzero ``(source, weight)`` pairs of each output pixel.

    Native pixel ``i`` is centered at ``scale * (i + 0.5)`` crop px,
    the placement the calibration fit scores with a zero origin.
    """
    reach = (len(taps) - 1) * spacing * scale
    rows = []
    for index in range(output):
        center = index + 0.5
        first = max(0, math.floor((center - reach) / scale - 0.5))
        last = min(native - 1, math.ceil((center + reach) / scale - 0.5))
        pairs = []
        for source in range(first, last + 1):
            weight = _tap_value(
                taps, spacing, (center - scale * (source + 0.5)) / scale
            )
            if weight:
                pairs.append((source, weight))
        rows.append(pairs)

    return rows


def resample_layers(
    layers: list[list[list[float]]],
    scale: float,
    kernel: SeparableKernel,
    size: tuple[int, int],
) -> list[list[list[float]]]:
    """Resample native layers (rows of values) to ``size`` (w, h).

    Each layer is filtered by the y taps along columns and the x taps
    along rows. Results keep the kernel's overshoot unclipped.
    """
    width, height = size
    native_height = len(layers[0])
    native_width = len(layers[0][0])
    weights_x = _axis_weights(
        kernel.x, kernel.spacing, native_width, width, scale
    )
    weights_y = _axis_weights(
        kernel.y, kernel.spacing, native_height, height, scale
    )
    results = []
    for layer in layers:
        horizontal = [
            [sum(row[i] * w for i, w in pairs) for pairs in weights_x]
            for row in layer
        ]
        results.append(
            [
                [
                    sum(horizontal[j][x] * w for j, w in pairs)
                    for x in range(width)
                ]
                for pairs in weights_y
            ]
        )

    return results
