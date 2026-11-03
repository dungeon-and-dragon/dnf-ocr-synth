"""Lay out a nickname the way the raid window's name cell draws it.

The renderer places Dotum Hangul with the font's 11px advance and
draws ASCII symbols with the Dotum face. In the raid window's name
cells every Hangul glyph instead starts at the cursor and advances its
ink width plus one pixel, the spacing the renderer already uses for
Dotumche ASCII, and ASCII symbols such as ``!``, ``-``, ``[`` and
``]`` are the Dotumche glyphs placed the same way. Both rules were
read off s=1 crops: ``!`` has Dotumche's two-row dot and ``-`` sits on
Dotumche's row, one lower than Dotum's. Other symbols keep the Dotum
glyph but sit one row lower than the font places them, with the same
cursor placement and ink advance; that rule was fitted on the scaled
``♥`` row of the third capture round (no s=1 capture of it exists),
where it took the row's mismatch from 10.2% to 1.0%, and it has not
been checked on any other symbol. This module applies the rules for
the calibration fit without changing the renderer's default output.

The name cell is 86 native px wide (dfragon's
``dnf-raid-participants.ts``) and the text is centered in it on the
native grid: a text ``width`` px wide starts ``(86 - width) / 2`` px
in, and when that is a half the cell rounds it up or down. The
direction is the same for every row of one capture but differs between
captures without a pattern in the cell's fractional native position
(round 1: raid A always rounds up, raids B and C either way), so
``cell_offset`` takes it as an argument.
"""

import unicodedata
from collections.abc import Callable

from PIL import Image

from ..render import GlyphLayout, Renderer, _raster

ADVANCE_MODES = ("ink", "font")
HANGUL_FACE = "dotum"
SYMBOL_FACE = "dotumche"
SYMBOL_ROW_OFFSET = 1

# Raid window layout in native pixels, from dfragon's
# dnf-raid-participants.ts: rows of 21px start under an 18px heading,
# the name cell starts 182 right of the heading anchor and 3 below
# its row, and a dark separator line runs along every row boundary.
RAID_HEADING_HEIGHT = 18
RAID_ROW_SPACING = 21
RAID_NAME_LEFT = 182
RAID_NAME_TOP = RAID_HEADING_HEIGHT + 3
RAID_NAME_WIDTH = 86
ROUNDINGS = ("up", "down")

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


def is_symbol(character: str) -> bool:
    """Tell whether ``character`` is a non-ASCII Unicode symbol."""
    return not character.isascii() and unicodedata.category(
        character
    ).startswith("S")


def ink_advance_layout(
    text: str, glyphs: GlyphLayout, symbol_raster: SymbolRaster
) -> GlyphLayout:
    """Return the layout with raid-window Hangul and symbol spacing.

    Hangul glyphs of the Dotum face start at the cursor and advance by
    ink width + 1. ASCII symbols take ``symbol_raster(character)``, the
    Dotumche mask with its baseline offset, placed the same way. Other
    symbols keep the Dotum mask, placed the same way but
    ``SYMBOL_ROW_OFFSET`` rows lower. Other glyphs keep their bearing
    and advance relative to the cursor, and every other vertical
    position is kept.
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
        elif name == HANGUL_FACE and is_symbol(character):
            positions.append((cursor, y + SYMBOL_ROW_OFFSET))
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


def text_box(glyphs: GlyphLayout) -> tuple[int, int, int]:
    """Return the ink box of the layout as ``(left, top, width)``.

    ``left`` and ``top`` are the smallest glyph positions, baseline
    relative like the positions themselves, and ``width`` spans every
    glyph's ink.
    """
    left = min(x for x, _ in glyphs.positions)
    top = min(y for _, y in glyphs.positions)
    right = max(
        x + mask.width for mask, (x, _) in zip(glyphs.masks, glyphs.positions)
    )

    return left, top, right - left


def cell_offset(width: int, rounding: str) -> int:
    """Return where a ``width`` px text starts in the 86px name cell.

    ``rounding`` (``up`` or ``down``) settles the half that an odd
    ``width`` leaves; an even ``width`` ignores it.
    """
    if rounding not in ROUNDINGS:
        raise ValueError(f"rounding must be one of {ROUNDINGS}.")
    double = RAID_NAME_WIDTH - width
    if rounding == "up":
        return -(-double // 2)

    return double // 2
