"""Lay out a nickname the way the raid window's name cell draws it.

The renderer places Dotum Hangul with the font's 11px advance and
draws ASCII symbols with the Dotum face. In the raid window's name
cells every Hangul glyph instead starts at the cursor and advances its
ink width plus one pixel, the spacing the renderer already uses for
Dotumche ASCII, and ASCII symbols such as ``!``, ``-``, ``[`` and
``]`` are the Dotumche glyphs placed the same way. Both rules were
read off s=1 crops: ``!`` has Dotumche's two-row dot and ``-`` sits on
Dotumche's row, one lower than Dotum's. This module applies them for
the calibration fit without changing the renderer's default output.
"""

from collections.abc import Callable

from PIL import Image

from ..render import GlyphLayout, Renderer, _raster

ADVANCE_MODES = ("ink", "font")
HANGUL_FACE = "dotum"
SYMBOL_FACE = "dotumche"

SymbolRaster = Callable[[str], tuple[Image.Image, tuple[int, int]]]


def is_hangul(character: str) -> bool:
    """Tell whether ``character`` is a precomposed Hangul syllable."""
    return "가" <= character <= "힣"


def is_ascii_symbol(character: str) -> bool:
    """Tell whether ``character`` is printable ASCII but not alnum."""
    return (
        character.isascii()
        and character.isprintable()
        and not character.isalnum()
        and not character.isspace()
    )


def ink_advance_layout(
    text: str, glyphs: GlyphLayout, symbol_raster: SymbolRaster
) -> GlyphLayout:
    """Return the layout with raid-window Hangul and symbol spacing.

    Hangul glyphs of the Dotum face start at the cursor and advance by
    ink width + 1. ASCII symbols take ``symbol_raster(character)``, the
    Dotumche mask with its baseline offset, placed the same way. Other
    glyphs keep their bearing and advance relative to the cursor, and
    every vertical position is kept.
    """
    cursor = 0
    original_cursor = 0
    names = []
    masks = []
    positions = []
    advances = []
    for character, name, mask, (x, y), advance in zip(
        text, glyphs.names, glyphs.masks, glyphs.positions, glyphs.advances
    ):
        if name == HANGUL_FACE and is_ascii_symbol(character):
            name = SYMBOL_FACE
            mask, (_, y) = symbol_raster(character)
            positions.append((cursor, y))
            advances.append(mask.width + 1)
        elif name == HANGUL_FACE and is_hangul(character):
            positions.append((cursor, y))
            advances.append(mask.width + 1)
        else:
            positions.append((cursor + x - original_cursor, y))
            advances.append(advance)
        names.append(name)
        masks.append(mask)
        cursor += advances[-1]
        original_cursor += advance

    return GlyphLayout(names, masks, positions, advances)


def raid_layout(
    renderer: Renderer, text: str, profile: str, mode: str
) -> GlyphLayout:
    """Lay ``text`` out with ``mode`` (``ink`` or ``font``) spacing."""
    if mode not in ADVANCE_MODES:
        raise ValueError(f"advance mode must be one of {ADVANCE_MODES}.")

    glyphs = renderer.layout(text, profile=profile)
    if mode == "font":
        return glyphs

    def symbol_raster(character: str) -> tuple[Image.Image, tuple[int, int]]:
        return _raster(character, renderer._load(SYMBOL_FACE))

    return ink_advance_layout(text, glyphs, symbol_raster)
