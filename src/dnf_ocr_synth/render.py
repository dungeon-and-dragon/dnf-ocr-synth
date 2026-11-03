"""Render glyphs, then outline and scale complete nickname rasters."""

import json
import math
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, features
from PIL.PngImagePlugin import PngInfo

from .fonts import FontPaths, _Face, _load_face
from .nickname import ValidationResult, validate_nickname
from .scale_model import (
    ScaleModel,
    SeparableKernel,
    resample_layers,
    resolve_scale_model,
)

COLOR = (75, 209, 255)
PADDING = 2
# The measured kernel reaches 2.5 native px past the 1px outline.
KERNEL_PADDING = 3


@dataclass
class GlyphLayout:
    """Hold native glyph rasters on the baseline-relative grid."""

    names: list[str]
    masks: list[Image.Image]
    positions: list[tuple[int, int]]
    advances: list[int]


@dataclass
class Sample:
    """Keep the RGBA result, native ink mask, and rendering metadata."""

    image: Image.Image
    native_mask: Image.Image
    metadata: dict

    def save(self, path: str | Path) -> None:
        """Save a PNG with UTF-8 JSON metadata under 'dnf_ocr_synth'."""
        info = PngInfo()
        info.add_itxt(
            "dnf_ocr_synth", json.dumps(self.metadata, ensure_ascii=False)
        )
        self.image.save(path, format="PNG", pnginfo=info)


def _raster(text: str, face: _Face) -> tuple[Image.Image, tuple[int, int]]:
    left, top, right, bottom = face.font.getbbox(text, anchor="ls")
    canvas = Image.new(face.mode, (right - left + 8, bottom - top + 8))
    ImageDraw.Draw(canvas).text(
        (4 - left, 4 - top),
        text,
        font=face.font,
        anchor="ls",
        fill=1 if face.mode == "1" else 255,
    )
    box = canvas.getbbox()
    if box is None:
        raise ValueError(f"Font produced no visible ink for {text!r}.")
    mask = canvas.crop(box).convert("L")
    if face.bold_x:
        expanded = Image.new("L", (mask.width + 1, mask.height))
        shifted = expanded.copy()
        expanded.paste(mask, (0, 0))
        shifted.paste(mask, (1, 0))
        mask = ImageChops.lighter(expanded, shifted)
    return mask, (left + box[0] - 4, top + box[1] - 4)


def _compose(
    masks: list[Image.Image], positions: list[tuple[int, int]]
) -> Image.Image:
    left = min(x for x, _ in positions)
    top = min(y for _, y in positions)
    right = max(x + mask.width for mask, (x, _) in zip(masks, positions))
    bottom = max(y + mask.height for mask, (_, y) in zip(masks, positions))
    canvas = Image.new("L", (right - left, bottom - top))
    for mask, (x, y) in zip(masks, positions):
        layer = Image.new("L", canvas.size)
        layer.paste(mask, (x - left, y - top))
        canvas = ImageChops.lighter(canvas, layer)
    return canvas


def _validate_text_request(
    text: str, profile: str, layout: str
) -> ValidationResult:
    validation = validate_nickname(text)
    if not validation.is_valid:
        raise ValueError(validation.reason)
    if profile not in {"dotum", "nanum-neo"}:
        raise ValueError("profile must be 'dotum' or 'nanum-neo'.")
    if layout != "metrics":
        raise ValueError("layout must be 'metrics'.")
    return validation


def _colorize(
    mask: Image.Image, scale: float, color: tuple[int, int, int]
) -> Image.Image:
    # Padding holds the 1px square outline and a transparent margin.
    ink = Image.new("L", (mask.width + 2 * PADDING, mask.height + 2 * PADDING))
    ink.paste(mask, (PADDING, PADDING))
    alpha = ink.filter(ImageFilter.MaxFilter(3))
    # Premultiply ink by its coverage within the black outline.
    channels = [
        ink.point([round(i * c / 255) for i in range(256)]) for c in color
    ]
    image = Image.merge("RGBa", (*channels, alpha))
    if scale != 1:
        size = (
            math.ceil(image.width * scale),
            math.ceil(image.height * scale),
        )
        # Ceil adds room without changing the isotropic scale.
        image = image.transform(
            size,
            Image.Transform.AFFINE,
            (1 / scale, 0, 0, 0, 1 / scale, 0),
            resample=Image.Resampling.BILINEAR,
        )
    return image.convert("RGBA")


def _kernel_colorize(
    mask: Image.Image,
    scale: float,
    color: tuple[int, int, int],
    kernel: SeparableKernel,
) -> Image.Image:
    size = (mask.width + 2 * KERNEL_PADDING, mask.height + 2 * KERNEL_PADDING)
    ink = Image.new("L", size)
    ink.paste(mask, (KERNEL_PADDING, KERNEL_PADDING))
    outlined = ink.filter(ImageFilter.MaxFilter(3))
    if scale == 1:
        # The game draws s=1 text as the native raster itself.
        fill = Image.new("RGB", size, color)
        image = Image.new("RGB", size)
        image.paste(fill, mask=ink.point(lambda value: 255 * (value > 0)))
        image.putalpha(outlined.point(lambda value: 255 * (value > 0)))

        return image

    def binary_rows(layer: Image.Image) -> list[list[float]]:
        values = list(layer.get_flattened_data())

        return [
            [float(value > 0) for value in values[y : y + layer.width]]
            for y in range(0, len(values), layer.width)
        ]

    output = (math.ceil(size[0] * scale), math.ceil(size[1] * scale))
    coverage, filled = resample_layers(
        [binary_rows(outlined), binary_rows(ink)], scale, kernel, output
    )
    pixels = []
    for coverage_row, filled_row in zip(coverage, filled):
        for alpha, fill in zip(coverage_row, filled_row):
            alpha = min(alpha, 1.0)
            opacity = round(alpha * 255)
            if opacity <= 0:
                pixels.append((0, 0, 0, 0))
                continue
            pixels.append(
                (
                    *(
                        round(min(255.0, max(0.0, value * fill / alpha)))
                        for value in color
                    ),
                    opacity,
                )
            )
    image = Image.new("RGBA", output)
    image.putdata(pixels)

    return image


class Renderer:
    """Reuse loaded local fonts across nickname samples.

    Hanja and Dotum ASCII letters/digits use measured spacing.
    Other characters retain font advances. Arbitrary mixed-script
    nicknames have not been verified against the game.
    """

    def __init__(self, fonts: FontPaths) -> None:
        """Retain paths and lazily cache selected font faces."""
        self.fonts = fonts
        self._faces: dict[str, _Face] = {}

    def _load(self, name: str) -> _Face:
        if name not in self._faces:
            self._faces[name] = _load_face(self.fonts, name)
        return self._faces[name]

    def _select(self, character: str, profile: str) -> str:
        codepoint = ord(character)
        if 0x4E00 <= codepoint <= 0x9FFF or 0xF900 <= codepoint <= 0xFAFF:
            choices = ["hanja"]
        elif profile == "dotum":
            if character == "I":
                choices = ["uttum"]
            elif character.isascii() and character.isalnum():
                choices = ["dotumche"]
            else:
                choices = ["dotum"]
        else:
            choices = ["nanum", "gungsuh"]
        for name in choices:
            if codepoint in self._load(name).coverage:
                return name
        raise ValueError(
            f"No glyph for U+{ord(character):04X} in profile {profile}."
        )

    def layout(
        self, text: str, *, profile: str = "dotum", layout: str = "metrics"
    ) -> GlyphLayout:
        """Rasterize glyphs and place them before outline and scaling.

        Positions are baseline-relative native pixels, so callers can
        scale glyphs individually. Missing glyphs raise ValueError.
        """
        _validate_text_request(text, profile, layout)
        return self._layout(text, profile)

    def _layout(self, text: str, profile: str) -> GlyphLayout:
        names = [self._select(character, profile) for character in text]
        masks = []
        positions = []
        advances = []
        cursor = 0
        for character, name in zip(text, names):
            face = self._load(name)
            glyph, (x, y) = _raster(character, face)
            masks.append(glyph)
            if name in {"dotumche", "uttum"}:
                # Game ASCII spacing starts at the first ink column.
                x = 0
                advance = glyph.width + (3 if character == "1" else 1)
            elif name == "hanja":
                advance = glyph.width + 1 if profile == "dotum" else 11
            else:
                advance = round(face.font.getlength(character)) + face.bold_x
            positions.append((cursor + x, y))
            advances.append(advance)
            cursor += advance
        return GlyphLayout(names, masks, positions, advances)

    def render(
        self,
        text: str,
        *,
        profile: str = "dotum",
        layout: str = "metrics",
        scale: float = 1.0,
        color: tuple[int, int, int] = COLOR,
        scale_model: str | ScaleModel = "affine-bilinear",
    ) -> Sample:
        """Render a nickname with an RGB fill and a 1px black outline.

        Scale in (0, 16] applies after rasterization. Missing glyphs
        raise ValueError. The label is never normalized, trimmed, or
        changed through character replacement.
        Color contains three integer RGB values from 0 to 255.
        ``scale_model`` names a model of ``SCALE_MODELS`` or is a
        ``ScaleModel``; the default keeps the 0.1.2 output. A kernel
        model needs a scale of at least 1, where it was measured.
        """
        validation = _validate_text_request(text, profile, layout)
        model = resolve_scale_model(scale_model)
        if len(color) != 3 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 255
            for value in color
        ):
            raise ValueError(
                "color must contain three integers from 0 to 255."
            )
        if (
            isinstance(scale, bool)
            or not math.isfinite(scale)
            or not 0 < scale <= 16
        ):
            raise ValueError("scale must be finite and in (0, 16].")
        if model.kernel is not None and scale < 1:
            raise ValueError(f"{model.name} needs a scale of at least 1.")
        if model.glyph_layout == "raid-name":
            # calibration.layout imports this module, so load it late.
            from .calibration.layout import raid_layout

            glyphs = raid_layout(self, text, profile, "ink")
        else:
            glyphs = self._layout(text, profile)
        names = glyphs.names
        advances = glyphs.advances
        mask = _compose(glyphs.masks, glyphs.positions)
        if model.kernel is None:
            image = _colorize(mask, scale, color)
            padding = PADDING
            resampling = "bilinear-premultiplied"
        else:
            image = _kernel_colorize(mask, scale, color, model.kernel)
            padding = KERNEL_PADDING
            resampling = (
                "native" if scale == 1 else "separable-kernel-premultiplied"
            )
        inspected_fonts = dict.fromkeys(
            ["nanum", *names] if profile == "nanum-neo" else names
        )
        metadata = {
            "schema_version": 2,
            "renderer_version": "0.2.0",
            "text": text,
            "codepoints": [f"U+{ord(character):04X}" for character in text],
            "cp949_bytes": validation.byte_length,
            "profile": profile,
            "layout": layout,
            "scale": scale,
            "foreground_rgb": list(color),
            "outline_rgb": [0, 0, 0],
            "outline_px": 1,
            "padding_px": padding,
            "native_ink_size": list(mask.size),
            "image_size": list(image.size),
            "resampling": resampling,
            "scale_model": model.metadata(),
            "layout_engine": "Pillow BASIC",
            "pillow_version": Image.__version__,
            "freetype_version": features.version_module("freetype2"),
            "fonts": [
                dict(self._load(name).metadata) for name in inspected_fonts
            ],
            "glyphs": [
                {
                    "character": character,
                    "font": name,
                    "advance_px": advance,
                }
                for character, name, advance in zip(text, names, advances)
            ],
        }
        return Sample(image, mask, metadata)
