"""Run the hypothesis fit or the HP bar edge measurement.

    python -m dnf_ocr_synth.calibration --crops index.json
        --manifest manifest.json --gulim gulim.ttc --out result/
    python -m dnf_ocr_synth.calibration hp-edge --image full.png
        --x 76 --width 178 --y 49

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
from ..render import COLOR, Renderer
from .align import (
    SEARCH_RADIUS,
    best_candidate,
    estimate_text_origin,
    search_alignment,
)
from .metrics import SCREENS, Reference, edge_profile
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


def _parse_fit(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration",
        description="Fit scaling pipelines to captured nickname crops.",
    )
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
        "--color", nargs=3, type=int, metavar=("R", "G", "B"), default=COLOR
    )
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
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument(
        "--limit", type=int, help="Fit only the first N labeled crops"
    )
    args = parser.parse_args(argv)
    if any(not 0 <= value <= 255 for value in args.color):
        parser.error("--color values must be between 0 and 255")
    args.color = tuple(args.color)
    args.outlined = args.outline_px == "pipeline"

    return args


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
            glyphs = renderer.layout(sample.text, profile=args.profile)
        except (ValueError, FileNotFoundError) as error:
            skipped.append(Skipped(sample.id, f"렌더 실패: {error}"))
            continue
        reference = Reference.from_rgba(
            _load_crop(args.crops.parent / sample.file),
            sample.scale,
            sample.screen,
        )
        base = render_with_pipeline(
            glyphs.masks,
            glyphs.positions,
            sample.scale,
            args.color,
            DEFAULT_PIPELINE,
            args.outlined,
        )
        crop_text_origin = estimate_text_origin(
            reference, np.asarray(base.image), base.text_origin, args.color
        )
        for pipeline in pipelines:
            rendered = render_with_pipeline(
                glyphs.masks,
                glyphs.positions,
                sample.scale,
                args.color,
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
        "color": list(args.color),
        "outline_px": "pipeline" if args.outlined else 0,
        "crops": str(args.crops),
        "manifest": str(args.manifest),
    }
    write_report(args.out, fitted, fits, skipped, rankings, settings)
    _write_residuals(args, renderer, fitted, fits, rankings)
    print(args.out / "report.md")


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
        glyphs = renderer.layout(sample.text, profile=args.profile)
        reference = Reference.from_rgba(
            _load_crop(args.crops.parent / sample.file),
            sample.scale,
            sample.screen,
        )
        for entry in rankings.get(sample.bucket, []):
            fit = by_key[(sample.id, entry.pipeline)]
            rendered = render_with_pipeline(
                glyphs.masks,
                glyphs.positions,
                sample.scale,
                args.color,
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


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the fit (default) or the ``hp-edge`` subcommand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["hp-edge"]:
            run_hp_edge(_parse_hp_edge(arguments[1:]))
        else:
            run_fit(_parse_fit(arguments))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
