"""Predict where each name crop's text sits from the capture's layout.

The fitted alignment gives every crop its own sub-pixel origin. The
predicted origin instead follows the raid window layout: within one
capture the name cell's left edge is the same for every row, the rows
are ``RAID_ROW_SPACING`` native px apart and each text is centered in
the cell. What the layout cannot give, the capture's sub-pixel phase
(the cell's left edge and the first row's baseline in capture px) and
the rounding of a half-pixel centering, is taken from the capture's
fitted rows, so a predicted origin carries three numbers per capture
instead of two per crop.
"""

import statistics
from collections import defaultdict
from dataclasses import dataclass

from .kernel import NATIVE_PADDING
from .layout import RAID_NAME_WIDTH, RAID_ROW_SPACING, cell_offset


@dataclass(frozen=True)
class Placement:
    """Pair one crop's fitted origin with what the layout knows of it.

    ``crop_origin`` is the crop's top-left in the capture, ``box`` the
    text's ink box ``(left, top, width)`` from ``text_box`` and
    ``origin`` the fitted origin of the padded native layers in crop
    px.
    """

    sample: str
    case_id: str
    slot: int
    scale: float
    crop_origin: tuple[int, int]
    box: tuple[int, int, int]
    origin: tuple[float, float]

    @property
    def ink_left(self) -> float:
        """Return the capture x where the text's ink starts."""
        return (
            self.crop_origin[0] + self.origin[0] + self.scale * NATIVE_PADDING
        )

    @property
    def baseline(self) -> float:
        """Return the capture y of the text's baseline."""
        _, top, _ = self.box

        return (
            self.crop_origin[1]
            + self.origin[1]
            + self.scale * (NATIVE_PADDING - top)
        )

    @property
    def width(self) -> int:
        """Return the text's ink width."""
        return self.box[2]


@dataclass(frozen=True)
class CaptureAnchor:
    """Hold the three numbers a capture's predicted origins need."""

    case_id: str
    cell_left: float
    first_baseline: float
    rounding: str

    def origin(self, placement: Placement) -> tuple[float, float]:
        """Predict the padded layers' origin of one of the rows."""
        scale = placement.scale
        _, top, width = placement.box
        ink_left = self.cell_left + scale * cell_offset(width, self.rounding)
        baseline = self.first_baseline + scale * RAID_ROW_SPACING * (
            placement.slot - 1
        )

        return (
            ink_left - scale * NATIVE_PADDING - placement.crop_origin[0],
            baseline
            - scale * (NATIVE_PADDING - top)
            - placement.crop_origin[1],
        )


def _anchor(case_id: str, rows: list[Placement]) -> CaptureAnchor:
    even = [row for row in rows if row.width % 2 == 0]
    odd = [row for row in rows if row.width % 2 == 1]
    rounding = "up"
    cell_left = statistics.median(
        row.ink_left - row.scale * cell_offset(row.width, rounding)
        for row in (even or odd)
    )
    if even and odd:
        halves = statistics.median(
            (row.ink_left - cell_left) / row.scale
            - (RAID_NAME_WIDTH - row.width) / 2
            for row in odd
        )
        if halves < 0:
            rounding = "down"
    first_baseline = statistics.median(
        row.baseline - row.scale * RAID_ROW_SPACING * (row.slot - 1)
        for row in rows
    )

    return CaptureAnchor(case_id, cell_left, first_baseline, rounding)


def predict_origins(
    placements: list[Placement],
) -> tuple[dict[str, tuple[float, float]], list[CaptureAnchor]]:
    """Predict every placement's origin from its capture's anchor.

    The anchor's cell edge comes from the rows with an even ink width,
    whose centering has no half to round; the rounding comes from the
    odd rows' side of that edge, ``up`` when a capture has no even row
    to tell. Returns the origins by sample id and the anchors.
    """
    by_case: dict[str, list[Placement]] = defaultdict(list)
    for placement in placements:
        by_case[placement.case_id].append(placement)
    origins = {}
    anchors = []
    for case_id, rows in by_case.items():
        anchor = _anchor(case_id, rows)
        anchors.append(anchor)
        for row in rows:
            origins[row.sample] = anchor.origin(row)

    return origins, anchors
