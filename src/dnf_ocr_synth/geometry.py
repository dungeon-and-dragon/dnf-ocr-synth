"""Place raid-window name crops the way dfragon cuts them.

dfragon (``packages/lib/src/dnf-raid-participants.ts``) finds the
423x18 raid heading at frame position ``(x, y)`` and scale ``s`` and
cuts row ``k``'s name cell from the native box left 182, top
``18 + 21 * (k - 1) + 3``, right 268 and bottom 17 rows lower,
projected as ``round(x + native * s)`` with exact halves rounded to the
even pixel (``roundParticipantPixel``). At s=1 that is an 86x17 crop;
at other scales the width and height depend on where the heading
sits. Party frame crops are not covered: the training target moved to
the raid window on 2026-11-03 (dfragon #1141).

Inside the cell the game centers the native text: a text ``width`` px
wide starts ``(86 - width) / 2`` px in, and a half is rounded up or
down per capture (``calibration.layout.cell_offset``).
"""

import math
import sys
from dataclasses import dataclass

from .calibration.layout import (
    RAID_NAME_LEFT,
    RAID_NAME_TOP,
    RAID_NAME_WIDTH,
    RAID_ROW_SPACING,
    cell_offset,
)

RAID_ROWS = 12
RAID_NAME_HEIGHT = 17
# Float noise from scale refinement can move an exact half by a few
# ulps; dfragon allows 4 epsilon before treating it as a half.
HALF_TOLERANCE_EPSILONS = 4


@dataclass(frozen=True)
class CropGeometry:
    """Hold a crop box in frame pixels."""

    x: int
    y: int
    width: int
    height: int


def round_pixel(value: float) -> int:
    """Round like dfragon's ``roundParticipantPixel``."""
    floor = math.floor(value)
    tolerance = (
        HALF_TOLERANCE_EPSILONS * sys.float_info.epsilon * max(1, abs(value))
    )
    if abs(value - floor - 0.5) <= tolerance:
        return floor + abs(floor % 2)

    return math.floor(value + 0.5)


def name_crop_geometry(
    scale: float, row: int = 1, heading: tuple[float, float] = (0.0, 0.0)
) -> CropGeometry:
    """Return the name crop of raid ``row`` (1-12) in frame pixels.

    ``heading`` is the frame position of the raid heading's top-left
    corner, ``scale`` its detected scale.
    """
    if not 1 <= row <= RAID_ROWS:
        raise ValueError(f"row must be from 1 to {RAID_ROWS}.")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive.")

    x, y = heading
    top = RAID_NAME_TOP + RAID_ROW_SPACING * (row - 1)
    left = round_pixel(x + RAID_NAME_LEFT * scale)
    upper = round_pixel(y + top * scale)
    right = round_pixel(x + (RAID_NAME_LEFT + RAID_NAME_WIDTH) * scale)
    bottom = round_pixel(y + (top + RAID_NAME_HEIGHT) * scale)

    return CropGeometry(left, upper, right - left, bottom - upper)


def name_text_left(width: int, rounding: str = "up") -> int:
    """Return where a ``width`` px native text starts in the name cell.

    ``rounding`` (``up`` or ``down``) settles the half an odd width
    leaves; every capture so far measured has one direction for all
    of its rows, most often ``up``.
    """
    return cell_offset(width, rounding)
