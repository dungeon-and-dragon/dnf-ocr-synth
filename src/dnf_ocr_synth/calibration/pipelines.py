"""Enumerate glyph scaling hypotheses and render layouts with them."""

import itertools
import math
from dataclasses import asdict, dataclass

from PIL import Image, ImageChops, ImageFilter

from ..render import _compose

OUTLINE_STAGES = ("before", "after")
OUTLINE_WIDTHS = ("1", "round")
RESAMPLES = ("nearest", "bilinear", "bicubic")
ALPHA_MODES = ("premultiplied", "straight")
TEXEL_OFFSETS = (0.0, 0.5)
ROUNDINGS = ("none", "floor", "round")

_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}


def _round_half_up(value: float) -> int:
    # Python's round() is banker's rounding; D3D-style placement is not.
    return math.floor(value + 0.5)


_PLACEMENTS = {
    "none": lambda value: value,
    "floor": math.floor,
    "round": _round_half_up,
}


@dataclass(frozen=True)
class ScalePipeline:
    """Describe one way the game could scale an outlined name raster.

    ``outline_stage`` draws the black outline before or after scaling.
    ``outline_px`` is ``"1"`` or ``"round"`` (``round(scale)``) pixels
    at the stage where it is drawn. ``texel_offset`` shifts sampling
    by that many native texels, so ``0.5`` models the D3D9 half-texel
    error. The glyph roundings place each glyph at ``floor`` or
    ``round`` of its scaled native position instead of scaling the
    whole line as one raster (``none``).
    """

    outline_stage: str = "before"
    outline_px: str = "1"
    resample: str = "bilinear"
    alpha: str = "premultiplied"
    texel_offset: float = 0.0
    glyph_x_rounding: str = "none"
    glyph_y_rounding: str = "none"

    def __post_init__(self) -> None:
        """Reject values outside the enumerated hypothesis space."""
        choices = {
            "outline_stage": OUTLINE_STAGES,
            "outline_px": OUTLINE_WIDTHS,
            "resample": RESAMPLES,
            "alpha": ALPHA_MODES,
            "texel_offset": TEXEL_OFFSETS,
            "glyph_x_rounding": ROUNDINGS,
            "glyph_y_rounding": ROUNDINGS,
        }
        for field, allowed in choices.items():
            value = getattr(self, field)
            if value not in allowed:
                raise ValueError(f"{field} must be one of {allowed}.")

    @property
    def name(self) -> str:
        """Return a file-name safe identifier for this pipeline."""
        texel = "t05" if self.texel_offset else "t0"

        return "-".join(
            [
                self.outline_stage,
                f"o{self.outline_px}",
                self.resample,
                self.alpha,
                texel,
                f"x{self.glyph_x_rounding}",
                f"y{self.glyph_y_rounding}",
            ]
        )

    def outline_pixels(self, scale: float) -> int:
        """Return the outline thickness in pixels at its stage."""
        if self.outline_px == "1":
            return 1

        return max(1, _round_half_up(scale))

    def as_dict(self) -> dict:
        """Return the parameters as JSON-friendly values."""
        return asdict(self)


DEFAULT_PIPELINE = ScalePipeline()


def enumerate_pipelines() -> list[ScalePipeline]:
    """List every combination of the hypothesis space (432)."""
    return [
        ScalePipeline(*values)
        for values in itertools.product(
            OUTLINE_STAGES,
            OUTLINE_WIDTHS,
            RESAMPLES,
            ALPHA_MODES,
            TEXEL_OFFSETS,
            ROUNDINGS,
            ROUNDINGS,
        )
    ]


@dataclass(frozen=True)
class PipelineRender:
    """Pair a scaled RGBA image with where the ink origin landed."""

    image: Image.Image
    text_origin: tuple[int, int]


def _color_tables(color: tuple[int, int, int]) -> list[list[int]]:
    return [[round(i * c / 255) for i in range(256)] for c in color]


def _native_layer(
    mask: Image.Image,
    padding: int,
    outline: int,
    stage: str,
    tables: list[list[int]],
) -> Image.Image:
    ink = Image.new("L", (mask.width + 2 * padding, mask.height + 2 * padding))
    ink.paste(mask, (padding, padding))
    if stage == "after":
        return ink
    alpha = ink.filter(ImageFilter.MaxFilter(2 * outline + 1))

    return Image.merge("RGBa", (*[ink.point(t) for t in tables], alpha))


def _scale_unit(
    layer: Image.Image,
    scale: float,
    padding: int,
    margin: int,
    fraction: tuple[float, float],
    resample: str,
    texel_offset: float,
) -> Image.Image:
    box_width = layer.width - 2 * padding
    box_height = layer.height - 2 * padding
    size = (
        math.ceil(box_width * scale) + 2 * margin,
        math.ceil(box_height * scale) + 2 * margin,
    )
    if scale == 1 and texel_offset == 0 and fraction == (0.0, 0.0):
        return layer

    # Output pixel u samples input (u - margin - fraction) / scale +
    # padding, so the native box origin lands at margin + fraction.
    return layer.transform(
        size,
        Image.Transform.AFFINE,
        (
            1 / scale,
            0,
            padding - (margin + fraction[0]) / scale + texel_offset,
            0,
            1 / scale,
            padding - (margin + fraction[1]) / scale + texel_offset,
        ),
        resample=_RESAMPLING[resample],
    )


def render_with_pipeline(
    masks: list[Image.Image],
    positions: list[tuple[int, int]],
    scale: float,
    color: tuple[int, int, int],
    pipeline: ScalePipeline,
) -> PipelineRender:
    """Outline, color, and scale a glyph layout with one hypothesis.

    The native ink origin (the union box of all glyphs) lands at an
    integer canvas position, so sub-pixel phase comes only from the
    pipeline. At scale 1 the default pipeline matches ``_colorize``.
    """
    outline = pipeline.outline_pixels(scale)
    padding = outline + 1
    margin = math.ceil(padding * scale)
    composed = _compose(masks, positions)
    left = min(x for x, _ in positions)
    top = min(y for _, y in positions)
    canvas_size = (
        math.ceil(composed.width * scale) + 2 * margin,
        math.ceil(composed.height * scale) + 2 * margin,
    )
    if pipeline.glyph_x_rounding == pipeline.glyph_y_rounding == "none":
        units = [(composed, (0.0, 0.0))]
    else:
        place_x = _PLACEMENTS[pipeline.glyph_x_rounding]
        place_y = _PLACEMENTS[pipeline.glyph_y_rounding]
        units = [
            (mask, (place_x((x - left) * scale), place_y((y - top) * scale)))
            for mask, (x, y) in zip(masks, positions)
        ]
    tables = _color_tables(color)
    canvas = None
    for mask, (x, y) in units:
        layer = _native_layer(
            mask, padding, outline, pipeline.outline_stage, tables
        )
        whole_x = math.floor(x)
        whole_y = math.floor(y)
        scaled = _scale_unit(
            layer,
            scale,
            padding,
            margin,
            (x - whole_x, y - whole_y),
            pipeline.resample,
            pipeline.texel_offset,
        )
        placed = Image.new(scaled.mode, canvas_size)
        placed.paste(scaled, (whole_x, whole_y))
        canvas = (
            placed if canvas is None else ImageChops.lighter(canvas, placed)
        )
    if pipeline.outline_stage == "after":
        alpha = canvas.filter(ImageFilter.MaxFilter(2 * outline + 1))
        canvas = Image.merge(
            "RGBa", (*[canvas.point(t) for t in tables], alpha)
        )

    if pipeline.alpha == "premultiplied":
        image = canvas.convert("RGBA")
    else:
        image = Image.merge("RGBA", canvas.split())

    return PipelineRender(image, (margin, margin))
