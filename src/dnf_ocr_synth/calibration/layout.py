"""Re-space a glyph layout the way the raid window draws Hangul.

The renderer places Dotum Hangul with the font's 11px advance. In the
raid window's name cells every Hangul glyph instead starts at the
cursor and advances its ink width plus one pixel, the spacing the
renderer already uses for Dotumche ASCII. This module applies that
spacing for the calibration fit without changing the renderer's
default output.
"""

from ..render import GlyphLayout

INK_ADVANCE_FACES = frozenset({"dotum"})
ADVANCE_MODES = ("ink", "font")


def ink_advance_layout(glyphs: GlyphLayout) -> GlyphLayout:
    """Return a layout whose ``dotum`` glyphs advance by ink width + 1.

    Other glyphs keep their bearing and advance relative to the
    cursor, and every vertical position is kept.
    """
    cursor = 0
    original_cursor = 0
    positions = []
    advances = []
    for name, mask, (x, y), advance in zip(
        glyphs.names, glyphs.masks, glyphs.positions, glyphs.advances
    ):
        if name in INK_ADVANCE_FACES:
            positions.append((cursor, y))
            advances.append(mask.width + 1)
        else:
            positions.append((cursor + x - original_cursor, y))
            advances.append(advance)
        cursor += advances[-1]
        original_cursor += advance

    return GlyphLayout(glyphs.names, glyphs.masks, positions, advances)


def layout_with_advance(glyphs: GlyphLayout, mode: str) -> GlyphLayout:
    """Apply ``mode`` (``ink`` or ``font``) to a renderer layout."""
    if mode not in ADVANCE_MODES:
        raise ValueError(f"advance mode must be one of {ADVANCE_MODES}.")

    if mode == "ink":
        return ink_advance_layout(glyphs)

    return glyphs
