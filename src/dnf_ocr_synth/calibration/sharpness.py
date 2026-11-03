"""Tell blurred captures from sharp ones by their pure-black share.

A sharp capture of the raid window keeps many outline pixels at
exactly 0 in every channel after the layer is scaled, while a frame
the game stretched from another internal resolution leaves almost
none. The share is measured per capture over all its name crops, so
it does not depend on the kernel, the alignment or the labels.
"""

import numpy as np

# Over rounds 1 to 3 (116 captures) the share of exactly-0 pixels
# among those whose darkest channel is at most 12 was 0.0225 to
# 0.0699 on the 16 blurred captures and 0.1606 to 0.2474 on the
# sharp scaled ones (0.35 to 0.46 at s=1), so 0.10 sits inside the
# gap with margin on both sides.
PURE_BLACK_MAX_DARKEST = 12
SHARP_MIN_PURE_BLACK_RATIO = 0.10


def pure_black_ratio(crops: list[np.ndarray]) -> float:
    """Return the share of exactly-black pixels among the near-black.

    ``crops`` are (H, W, 3+) uint8 arrays of one capture. Raises
    ValueError when no pixel is near black, since the share is then
    undefined.
    """
    zero = 0
    near = 0
    for crop in crops:
        darkest = np.asarray(crop, dtype=np.uint8)[..., :3].min(axis=-1)
        zero += int((darkest == 0).sum())
        near += int((darkest <= PURE_BLACK_MAX_DARKEST).sum())
    if near == 0:
        raise ValueError("The crops have no near-black pixel.")

    return zero / near


def is_sharp(ratio: float) -> bool:
    """Tell whether a capture with this pure-black share is sharp."""
    return ratio >= SHARP_MIN_PURE_BLACK_RATIO
