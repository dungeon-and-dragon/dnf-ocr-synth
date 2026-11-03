"""Run the hypothesis fit or an edge measurement on full captures.

    python -m dnf_ocr_synth.calibration --crops index.json
        --manifest manifest.json --gulim gulim.ttc --out result/
    python -m dnf_ocr_synth.calibration hp-edge --image full.png
        --x 76 --width 178 --y 49
    python -m dnf_ocr_synth.calibration raid-lines --crops index.json
        --originals originals/

``index.json`` and ``manifest.json`` follow dfragon's
``nickname-evaluation`` script, which writes crops as
``{caseId}_slot{n}.png`` next to ``index.json`` and keeps the labeled
nicknames in the manifest. Both name the ``screen`` (``party`` or
``raid``) a crop came from; raid slots are the rows 1 to 12.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..fonts import FontPaths
from ..render import COLOR, GlyphLayout, Renderer
from .align import (
    SEARCH_RADIUS,
    best_candidate,
    estimate_text_origin,
    nearest_fill_color,
    search_alignment,
)
from .layout import ADVANCE_MODES, layout_with_advance
from .measured import run_measured_kernel
from .metrics import (
    SCREENS,
    Reference,
    edge_profile,
    line_profile,
    line_width,
)
from .pipelines import (
    DEFAULT_PIPELINE,
    enumerate_pipelines,
    render_with_pipeline,
)
from .report import (
    Fit,
    Ranking,
    SampleInfo,
    Skipped,
    rank_pipelines,
    residual_image,
    write_report,
)


def _require(mapping: dict, key: str, where: str):
    if key not in mapping:
        raise ValueError(f"{where} lacks {key!r}.")

    return mapping[key]


def _require_screen(mapping: dict, where: str) -> str:
    screen = _require(mapping, "screen", where)
    if screen not in SCREENS:
        raise ValueError(f"{where} screen must be one of {SCREENS}.")

    return screen


def load_samples(
    crops_index: Path, manifest: Path, include_uncertain: bool
) -> tuple[list[SampleInfo], list[Skipped]]:
    """Join crop index entries with manifest labels.

    Crops without a label are skipped, and uncertain labels are
    skipped unless ``include_uncertain`` is set. Every crop entry and
    manifest case must name its ``screen``, and the two must agree, so
    an index written before screens existed is rejected instead of
    being read as party crops.
    """
    index = json.loads(crops_index.read_text(encoding="utf-8"))
    labels = {}
    cases = _require(
        json.loads(manifest.read_text(encoding="utf-8")), "cases", "manifest"
    )
    for case in cases:
        case_id = _require(case, "id", "manifest case")
        screen = _require_screen(case, "manifest case")
        for slot in _require(case, "slots", "manifest case"):
            labels[(case_id, _require(slot, "slot", "manifest slot"))] = (
                screen,
                _require(slot, "text", "manifest slot"),
                bool(slot.get("uncertain", False)),
            )
    samples = []
    skipped = []
    for position, entry in enumerate(_require(index, "crops", "crop index")):
        where = f"crops[{position}]"
        case_id = _require(entry, "caseId", where)
        screen = _require_screen(entry, where)
        slot = _require(entry, "slot", where)
        sample_id = f"{case_id}_slot{slot}"
        label = labels.get((case_id, slot))
        if label is None:
            skipped.append(Skipped(sample_id, "manifest에 정답이 없음"))
            continue
        case_screen, text, uncertain = label
        if case_screen != screen:
            raise ValueError(
                f"{where} screen {screen!r} differs from manifest case "
                f"{case_id!r} screen {case_screen!r}."
            )

        if uncertain and not include_uncertain:
            skipped.append(Skipped(sample_id, "정답이 uncertain"))
            continue
        samples.append(
            SampleInfo(
                sample_id,
                case_id,
                screen,
                slot,
                _require(entry, "file", where),
                float(_require(entry, "scale", where)),
                text,
                entry.get("uiPercent"),
                str(entry.get("capture", "")),
            )
        )

    return samples, skipped


def _add_sample_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gulim", type=Path)
    parser.add_argument("--batang", type=Path)
    parser.add_argument("--nanum", type=Path)
    parser.add_argument("--uttum", type=Path)
    parser.add_argument(
        "--profile", choices=("dotum", "nanum-neo"), default="dotum"
    )
    parser.add_argument(
        "--color",
        nargs="+",
        metavar="R G B | auto",
        default=[str(value) for value in COLOR],
        help=(
            "Fill color, or 'auto' to pick the raid fill color nearest "
            "to each crop's bright pixels"
        ),
    )
    parser.add_argument(
        "--advance",
        choices=ADVANCE_MODES,
        default="ink",
        help=(
            "Hangul spacing: ink width + 1 as the raid window draws it "
            "(default) or the renderer's font advance"
        ),
    )
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Fit only the first N labeled crops"
    )


def _parse_fit(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration",
        description="Fit scaling pipelines to captured nickname crops.",
    )
    _add_sample_arguments(parser)
    parser.add_argument(
        "--outline-px",
        choices=("pipeline", "0"),
        default="pipeline",
        help=(
            "Outline thickness: each pipeline's own (default) or 0 for "
            "name cells drawn without an outline"
        ),
    )
    parser.add_argument("--radius", type=int, default=SEARCH_RADIUS)
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args(argv)
    args.color = _parse_color(parser, args.color)
    args.outlined = args.outline_px == "pipeline"

    return args


def _parse_measured(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration measured-kernel",
        description=(
            "Estimate one separable resampling kernel per scale bucket "
            "from the crops and score every crop with it."
        ),
    )
    _add_sample_arguments(parser)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args(argv)
    args.color = _parse_color(parser, args.color)

    return args


def run_measured(args: argparse.Namespace) -> None:
    """Fit the measured kernel per bucket and write its report."""
    samples, skipped = load_samples(
        args.crops, args.manifest, args.include_uncertain
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    renderer = Renderer(
        FontPaths(
            gulim=args.gulim,
            batang=args.batang,
            nanum=args.nanum,
            uttum=args.uttum,
        )
    )
    settings = {
        "iterations": args.iterations,
        "profile": args.profile,
        "color": "auto" if args.color is None else list(args.color),
        "advance": args.advance,
        "crops": str(args.crops),
        "manifest": str(args.manifest),
    }
    run_measured_kernel(
        args.out,
        samples,
        skipped,
        lambda sample: _prepare_sample(args, renderer, sample),
        args.iterations,
        settings,
    )
    print(args.out / "measured-kernel.md")


def _parse_color(
    parser: argparse.ArgumentParser, values: list[str]
) -> tuple[int, int, int] | None:
    """Return the RGB triple, or None when the color is ``auto``."""
    if values == ["auto"]:
        return None

    if len(values) != 3 or not all(value.isdigit() for value in values):
        parser.error("--color takes three integers or 'auto'")
    color = tuple(int(value) for value in values)
    if any(not 0 <= value <= 255 for value in color):
        parser.error("--color values must be between 0 and 255")

    return color


def _sample_color(
    color: tuple[int, int, int] | None, crop: np.ndarray
) -> tuple[int, int, int]:
    if color is None:
        return nearest_fill_color(crop)

    return color


def _load_crop(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))


def run_fit(args: argparse.Namespace) -> None:
    """Fit every pipeline to every labeled crop and write the report."""
    samples, skipped = load_samples(
        args.crops, args.manifest, args.include_uncertain
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    renderer = Renderer(
        FontPaths(
            gulim=args.gulim,
            batang=args.batang,
            nanum=args.nanum,
            uttum=args.uttum,
        )
    )
    pipelines = enumerate_pipelines()
    fits: list[Fit] = []
    fitted: list[SampleInfo] = []
    started = time.monotonic()
    for index, sample in enumerate(samples, 1):
        try:
            glyphs, reference, color = _prepare_sample(args, renderer, sample)
        except (ValueError, FileNotFoundError) as error:
            skipped.append(Skipped(sample.id, f"렌더 실패: {error}"))
            continue
        base = render_with_pipeline(
            glyphs.masks,
            glyphs.positions,
            sample.scale,
            color,
            DEFAULT_PIPELINE,
            args.outlined,
        )
        crop_text_origin = estimate_text_origin(
            reference, np.asarray(base.image), base.text_origin, color
        )
        for pipeline in pipelines:
            rendered = render_with_pipeline(
                glyphs.masks,
                glyphs.positions,
                sample.scale,
                color,
                pipeline,
                args.outlined,
            )
            best = best_candidate(
                search_alignment(
                    reference,
                    np.asarray(rendered.image),
                    rendered.text_origin,
                    crop_text_origin,
                    args.radius,
                )
            )
            fits.append(
                Fit(
                    sample.id,
                    pipeline.name,
                    best.offset,
                    best.origin,
                    best.comparison.mae,
                    best.comparison.mismatch,
                    best.comparison.ink,
                )
            )
        fitted.append(sample)
        print(
            f"[{index}/{len(samples)}] {sample.id} s={sample.scale:.2f} "
            f"{time.monotonic() - started:.0f}s",
            file=sys.stderr,
        )
    rankings = rank_pipelines(fitted, fits, args.top)
    settings = {
        "pipeline_count": len(pipelines),
        "radius": args.radius,
        "top": args.top,
        "profile": args.profile,
        "color": "auto" if args.color is None else list(args.color),
        "advance": args.advance,
        "outline_px": "pipeline" if args.outlined else 0,
        "crops": str(args.crops),
        "manifest": str(args.manifest),
    }
    write_report(args.out, fitted, fits, skipped, rankings, settings)
    _write_residuals(args, renderer, fitted, fits, rankings)
    print(args.out / "report.md")


def _prepare_sample(
    args: argparse.Namespace, renderer: Renderer, sample: SampleInfo
) -> tuple[GlyphLayout, Reference, tuple[int, int, int]]:
    glyphs = layout_with_advance(
        renderer.layout(sample.text, profile=args.profile), args.advance
    )
    crop = _load_crop(args.crops.parent / sample.file)
    reference = Reference.from_rgba(crop, sample.scale, sample.screen)

    return glyphs, reference, _sample_color(args.color, crop)


def _write_residuals(
    args: argparse.Namespace,
    renderer: Renderer,
    samples: list[SampleInfo],
    fits: list[Fit],
    rankings: dict[str, list[Ranking]],
) -> None:
    residual_dir = args.out / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    by_name = {pipeline.name: pipeline for pipeline in enumerate_pipelines()}
    by_key = {(fit.sample, fit.pipeline): fit for fit in fits}
    for sample in samples:
        glyphs, reference, color = _prepare_sample(args, renderer, sample)
        for entry in rankings.get(sample.bucket, []):
            fit = by_key[(sample.id, entry.pipeline)]
            rendered = render_with_pipeline(
                glyphs.masks,
                glyphs.positions,
                sample.scale,
                color,
                by_name[entry.pipeline],
                args.outlined,
            )
            residual_image(
                reference, np.asarray(rendered.image), fit.origin
            ).save(residual_dir / f"{sample.id}_{entry.pipeline}.png")


def _parse_hp_edge(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration hp-edge",
        description=(
            "Measure how far the HP bar's top edge spreads over rows "
            "to tell whole-layer scaling from crisp bar drawing."
        ),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--x", type=int, required=True, help="Bar left")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument(
        "--y", type=int, required=True, help="Approximate top edge row"
    )
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument(
        "--channel", choices=("red", "green", "blue"), default="red"
    )

    return parser.parse_args(argv)


def run_hp_edge(args: argparse.Namespace) -> None:
    """Print the edge profile as JSON."""
    with Image.open(args.image) as image:
        pixels = np.asarray(image.convert("RGB"))
    profile = edge_profile(
        pixels,
        args.x,
        args.width,
        args.y,
        args.window,
        ("red", "green", "blue").index(args.channel),
    )
    print(json.dumps(asdict(profile), ensure_ascii=False))


# Raid window layout in native pixels, from dfragon's
# dnf-raid-participants.ts: rows of 21px start under an 18px heading,
# the name cell starts 182 right of the heading anchor and 3 below
# its row, and a dark separator line runs along every row boundary.
RAID_HEADING_HEIGHT = 18
RAID_ROW_SPACING = 21
RAID_NAME_LEFT = 182
RAID_NAME_TOP = RAID_HEADING_HEIGHT + 3
RAID_NAME_WIDTH = 86
RAID_SEPARATOR_LINES = range(1, 12)


def _parse_raid_lines(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration raid-lines",
        description=(
            "Measure the raid window's row separator lines on the "
            "full captures to tell whole-layer scaling from crisp "
            "redrawing: a native 2px line scaled with the layer "
            "measures about 2s rows wide."
        ),
    )
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument(
        "--originals",
        type=Path,
        required=True,
        help="Folder holding {caseId}.png full captures",
    )
    parser.add_argument("--window", type=int, default=3)

    return parser.parse_args(argv)


def raid_separator_profiles(
    index: dict, originals: Path, window: int
) -> list[dict]:
    """Measure the separator lines under the name column per case.

    The heading anchor is recovered from each case's row 1 crop, and
    the darkest-channel profile is averaged over the lines between
    rows 1 to 12 so a single name's ink does not dominate.
    """
    results = []
    for entry in _require(index, "crops", "crop index"):
        if _require(entry, "slot", "crop entry") != 1:
            continue

        case_id = _require(entry, "caseId", "crop entry")
        scale = float(_require(entry, "scale", "crop entry"))
        crop = _require(entry, "crop", "crop entry")
        anchor_x = (
            _require(crop, "x", "crop rectangle") - RAID_NAME_LEFT * scale
        )
        anchor_y = (
            _require(crop, "y", "crop rectangle") - RAID_NAME_TOP * scale
        )
        with Image.open(originals / f"{case_id}.png") as image:
            darkest = np.asarray(image.convert("RGB")).min(axis=2)[..., None]
        x = round(anchor_x + (RAID_NAME_LEFT + 1) * scale)
        width = round((RAID_NAME_WIDTH - 2) * scale)
        values = np.mean(
            [
                line_profile(
                    darkest,
                    x,
                    width,
                    round(
                        anchor_y
                        + (RAID_HEADING_HEIGHT + RAID_ROW_SPACING * line)
                        * scale
                    ),
                    window,
                ).values
                for line in RAID_SEPARATOR_LINES
            ],
            axis=0,
        ).tolist()
        results.append(
            {
                "caseId": case_id,
                "scale": scale,
                "values": [round(value, 2) for value in values],
                "width_rows": line_width(values),
            }
        )

    return results


def run_raid_lines(args: argparse.Namespace) -> None:
    """Print one JSON object per case with the separator line width."""
    index = json.loads(args.crops.read_text(encoding="utf-8"))
    for result in raid_separator_profiles(index, args.originals, args.window):
        print(json.dumps(result, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the fit (default) or a measurement subcommand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["hp-edge"]:
            run_hp_edge(_parse_hp_edge(arguments[1:]))
        elif arguments[:1] == ["raid-lines"]:
            run_raid_lines(_parse_raid_lines(arguments[1:]))
        elif arguments[:1] == ["measured-kernel"]:
            run_measured(_parse_measured(arguments[1:]))
        else:
            run_fit(_parse_fit(arguments))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
