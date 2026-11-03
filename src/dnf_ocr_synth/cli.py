"""Parse command-line options for nickname image generation."""

import argparse
from pathlib import Path

from .nickname import estimate_ui_scale
from .render import COLOR
from .scale_model import SCALE_MODELS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI options and resolve the requested image scale."""
    parser = argparse.ArgumentParser(
        description="Generate one PNG from the command line."
    )
    parser.add_argument(
        "text", help="Original CP949 nickname (up to 12 bytes)"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=("dotum", "nanum-neo"), default="dotum"
    )
    parser.add_argument("--layout", choices=("metrics",), default="metrics")
    parser.add_argument(
        "--color",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        default=COLOR,
        help="Text color as RGB values (0-255; default: 75 209 255)",
    )
    parser.add_argument("--gulim", type=Path)
    parser.add_argument(
        "--batang",
        type=Path,
        help="Font for Hanja in both profiles and nanum-neo fallback",
    )
    parser.add_argument("--nanum", type=Path)
    parser.add_argument(
        "--uttum",
        type=Path,
        help="UttumBatangBold.ttf for capital I in the dotum profile",
    )
    scaling = parser.add_mutually_exclusive_group()
    scaling.add_argument("--scale", type=float)
    scaling.add_argument("--ui-percent", type=float)
    parser.add_argument("--client-height", type=int, default=1080)
    parser.add_argument(
        "--scale-model",
        choices=tuple(SCALE_MODELS),
        default="affine-bilinear",
        help=(
            "Scaling method: the 0.1.2 bilinear transform (default) or "
            "the kernel and raid name layout measured in the game"
        ),
    )
    args = parser.parse_args(argv)
    if any(not 0 <= value <= 255 for value in args.color):
        parser.error("--color values must be between 0 and 255")
    args.color = tuple(args.color)
    if args.ui_percent is not None:
        try:
            args.scale = estimate_ui_scale(args.ui_percent, args.client_height)
        except ValueError as error:
            parser.exit(2, f"error: {error}\n")
    elif args.client_height != 1080:
        parser.error("--client-height requires --ui-percent")
    elif args.scale is None:
        args.scale = 1.0
    return args
