"""Place synthetic renders on crops and pick the best offset."""

from dataclasses import dataclass

import numpy as np

from .metrics import Comparison, Reference, bt601_gray, compare

SEARCH_RADIUS = 2
# Fill colors measured on s=1 raid name cells: plain rows and glow
# rows (the viewer's own party), whose name is drawn translucently over
# a golden glow.
RAID_FILL_COLORS = {"beige": (214, 197, 146), "yellow": (255, 255, 147)}
PLAIN_FILL = RAID_FILL_COLORS["beige"]
GLOW_FILL = RAID_FILL_COLORS["yellow"]
# The glow tints the dark background: over the 648 round-1 crops the
# dark pixels' mean red minus blue was at most 2.4 on plain rows and
# at least 27.7 on glow rows, while the blended fill of scaled glow
# rows gets close enough to beige to fool a fill-color distance.
BACKGROUND_MAX_CHANNEL = 80
GLOW_MIN_RED_MINUS_BLUE = 15.0


def is_glow_row(rgba: np.ndarray) -> bool:
    """Tell a glow row from a plain row by its background tint.

    Pixels whose brightest channel stays below
    ``BACKGROUND_MAX_CHANNEL`` are background; a crop without any
    raises ValueError.
    """
    pixels = np.asarray(rgba, dtype=np.uint8).reshape(-1, 4)[:, :3]
    dark = pixels[pixels.max(axis=1) < BACKGROUND_MAX_CHANNEL].astype(
        np.float64
    )
    if dark.size == 0:
        raise ValueError("The crop has no background pixels.")
    red, _, blue = dark.mean(axis=0)

    return red - blue >= GLOW_MIN_RED_MINUS_BLUE


def raid_fill_color(rgba: np.ndarray) -> tuple[int, int, int]:
    """Return the fill color of a raid name crop from its row kind."""
    if is_glow_row(rgba):
        return GLOW_FILL

    return PLAIN_FILL


def fill_map(synthetic: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    """Approximate which synthetic pixels the product would read as ink.

    The synthetic is composited over black and kept where the luma
    reaches half of the fill color's luma, which separates fill from
    the black outline and faint edges.
    """
    pixels = np.asarray(synthetic, dtype=np.int64)
    over_black = pixels[..., :3] * pixels[..., 3:4] // 255
    threshold = int(bt601_gray(np.array(color, dtype=np.uint8))) // 2

    return bt601_gray(over_black) >= threshold


def estimate_text_origin(
    reference: Reference,
    synthetic: np.ndarray,
    text_origin: tuple[int, int],
    color: tuple[int, int, int],
) -> tuple[int, int]:
    """Find where the native ink origin sits in crop coordinates.

    Every placement of the synthetic fill over the crop ink map is
    scored by overlap; ties keep the first in raster order.
    """
    fill = fill_map(synthetic, color)
    rows = np.flatnonzero(fill.any(axis=1))
    columns = np.flatnonzero(fill.any(axis=0))
    if rows.size == 0:
        raise ValueError("The synthetic render has no fill pixels.")
    box = fill[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    ink = reference.ink
    height, width = ink.shape
    box_height, box_width = box.shape
    padded = np.zeros(
        (height + 2 * box_height - 2, width + 2 * box_width - 2), dtype=bool
    )
    padded[
        box_height - 1 : box_height - 1 + height,
        box_width - 1 : box_width - 1 + width,
    ] = ink
    best = (-1, 0, 0)
    for y in range(padded.shape[0] - box_height + 1):
        window = padded[y : y + box_height]
        for x in range(padded.shape[1] - box_width + 1):
            overlap = int((window[:, x : x + box_width] & box).sum())
            if overlap > best[0]:
                best = (overlap, x, y)
    _, x, y = best
    box_origin = (x - (box_width - 1), y - (box_height - 1))

    return (
        box_origin[0] - int(columns[0]) + text_origin[0],
        box_origin[1] - int(rows[0]) + text_origin[1],
    )


@dataclass(frozen=True)
class Candidate:
    """Hold one tried offset with the resulting comparison."""

    offset: tuple[int, int]
    origin: tuple[int, int]
    comparison: Comparison

    def sort_key(self) -> tuple:
        """Prefer fewer Otsu mismatches, lower MAE, then no shift."""
        dx, dy = self.offset

        return (
            self.comparison.mismatch,
            self.comparison.mae,
            abs(dx) + abs(dy),
            dx,
            dy,
        )


def search_alignment(
    reference: Reference,
    synthetic: np.ndarray,
    synthetic_text_origin: tuple[int, int],
    crop_text_origin: tuple[int, int],
    radius: int = SEARCH_RADIUS,
) -> list[Candidate]:
    """Compare every integer offset within ``radius`` of the base."""
    base_x = crop_text_origin[0] - synthetic_text_origin[0]
    base_y = crop_text_origin[1] - synthetic_text_origin[1]
    candidates = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            origin = (base_x + dx, base_y + dy)
            candidates.append(
                Candidate(
                    (dx, dy), origin, compare(reference, synthetic, origin)
                )
            )

    return candidates


def best_candidate(candidates: list[Candidate]) -> Candidate:
    """Pick the deterministic best of ``search_alignment``."""
    return min(candidates, key=Candidate.sort_key)
