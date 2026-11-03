"""Run the hypothesis fit or an edge measurement on full captures.

    python -m dnf_ocr_synth.calibration --crops index.json
        --manifest manifest.json --gulim gulim.ttc --out result/
    python -m dnf_ocr_synth.calibration hp-edge --image full.png
        --x 76 --width 178 --y 49
    python -m dnf_ocr_synth.calibration raid-lines --crops index.json
        --originals originals/
    python -m dnf_ocr_synth.calibration phase --lines raid-lines.jsonl
        --measured result/measured-kernel.json --crops index.json
    python -m dnf_ocr_synth.calibration sharpness --crops index.json

``index.json`` and ``manifest.json`` follow dfragon's
``nickname-evaluation`` script, which writes crops as
``{caseId}_slot{n}.png`` next to ``index.json`` and keeps the labeled
nicknames in the manifest. Both name the ``screen`` (``party`` or
``raid``) a crop came from; raid slots are the rows 1 to 12.
"""

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from ..fonts import FontPaths
from ..render import COLOR, GlyphLayout, Renderer
from ..scale_model import SCALE_MODELS, ScaleModel
from .align import (
    SEARCH_RADIUS,
    best_candidate,
    estimate_text_origin,
    raid_fill_color,
    search_alignment,
)
from .kernel import TAP_SPACING, KernelTaps
from .layout import (
    ADVANCE_MODES,
    RAID_HEADING_HEIGHT,
    RAID_NAME_LEFT,
    RAID_NAME_TOP,
    RAID_NAME_WIDTH,
    RAID_ROW_SPACING,
    raid_layout,
)
from .measured import (
    CANDIDATES,
    CAPTURE_SETS,
    DEFAULT_RESIDUALS,
    HOLDOUTS,
    ORIGIN_MODES,
    SCALE_SOURCES,
    FittedKernel,
    load_kernels,
    run_measured_kernel,
)
from .metrics import (
    SCREENS,
    Reference,
    edge_profile,
    line_center,
    line_profile,
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
from .sharpness import is_sharp, pure_black_ratio

KERNEL_SCALE_MODELS = tuple(
    name for name, model in SCALE_MODELS.items() if model.kernel is not None
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
    raids = {}
    cases = _require(
        json.loads(manifest.read_text(encoding="utf-8")), "cases", "manifest"
    )
    for case in cases:
        case_id = _require(case, "id", "manifest case")
        screen = _require_screen(case, "manifest case")
        raids[case_id] = str(case.get("raid", ""))
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
                str(crops_index.parent / _require(entry, "file", where)),
                float(_require(entry, "scale", where)),
                text,
                entry.get("uiPercent"),
                str(entry.get("capture", "")),
                raids[case_id],
                _theory_scale(entry),
                _crop_origin(entry),
            )
        )

    return samples, skipped


def _crop_origin(entry: dict) -> tuple[int, int] | None:
    """Return the crop's top-left in the capture, if the entry says."""
    crop = entry.get("crop")
    if crop is None:
        return None

    return (int(crop["x"]), int(crop["y"]))


def _theory_scale(entry: dict) -> float | None:
    """Return the scale the UI setting predicts, if the entry says.

    This is dfragon's 600px model behind ``estimate_ui_scale``, applied
    here without its client-height limits because captures of a
    window taller than 1080 rows are still calibrated.
    """
    client = entry.get("client")
    ui_percent = entry.get("uiPercent")
    if client is None or ui_percent is None:
        return None
    height = int(client["height"])

    return height / (height - (height - 600) * ui_percent / 100)


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
            "Fill color, or 'auto' to pick the raid fill color by each "
            "crop's row kind (plain or glow background)"
        ),
    )
    parser.add_argument(
        "--advance",
        choices=ADVANCE_MODES,
        default="ink",
        help=(
            "Spacing: Hangul at ink width + 1 and ASCII symbols as "
            "Dotumche glyphs, as the raid window draws them (default), "
            "or the renderer's font layout"
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
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=tuple(CANDIDATES),
        default=list(CANDIDATES),
        metavar="NAME",
        help=f"Kernels to compare, from {', '.join(CANDIDATES)}",
    )
    parser.add_argument(
        "--holdouts",
        nargs="*",
        choices=HOLDOUTS,
        default=list(HOLDOUTS),
        help="Hold-out schemes for the fitted candidates (default: both)",
    )
    parser.add_argument(
        "--residuals",
        nargs="*",
        choices=tuple(CANDIDATES),
        default=list(DEFAULT_RESIDUALS),
        metavar="NAME",
        help="Candidates whose per-sample tables and sheets are written",
    )
    parser.add_argument(
        "--scale-source",
        choices=SCALE_SOURCES,
        default="detected",
        help=(
            "Resample with the crop's detected scale (default) or the "
            "scale the client height and UI setting predict"
        ),
    )
    parser.add_argument(
        "--fit-crops",
        type=Path,
        help=(
            "Estimate parameters from this crop index instead and score "
            "every --crops crop as hold-out"
        ),
    )
    parser.add_argument(
        "--fit-manifest", type=Path, help="Manifest of --fit-crops"
    )
    parser.add_argument(
        "--kernels",
        type=Path,
        help=(
            "Score every --crops crop with the kernels of this "
            "kernels.json instead of fitting anything"
        ),
    )
    parser.add_argument(
        "--scale-model",
        choices=KERNEL_SCALE_MODELS,
        help=(
            "Score measured-native with this packaged scale model's "
            "kernel instead of fitting anything"
        ),
    )
    parser.add_argument(
        "--origins",
        nargs="+",
        choices=ORIGIN_MODES,
        default=list(ORIGIN_MODES),
        help="Origin modes to score (default: both)",
    )
    parser.add_argument(
        "--captures",
        choices=CAPTURE_SETS,
        default="all",
        help="Score every capture or only the sharp ones (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Alignment processes"
    )
    args = parser.parse_args(argv)
    args.color = _parse_color(parser, args.color)
    if (args.fit_crops is None) != (args.fit_manifest is None):
        parser.error("--fit-crops and --fit-manifest go together")
    if args.kernels is not None and args.fit_crops is not None:
        parser.error("--kernels and --fit-crops exclude each other")
    if args.scale_model is not None:
        if args.kernels is not None or args.fit_crops is not None:
            parser.error("--scale-model excludes --kernels and --fit-crops")
        if args.advance != "ink":
            parser.error("--scale-model lays names out with --advance ink")
    for name in args.residuals:
        if name not in args.candidates:
            parser.error(f"--residuals {name} is not among --candidates")
    for position, name in enumerate(args.candidates):
        base = CANDIDATES[name].base
        if base is not None and base not in args.candidates[:position]:
            parser.error(f"--candidates {name} needs {base} before it")

    return args


def run_measured(args: argparse.Namespace) -> None:
    """Compare the candidate kernels and write the report."""
    samples, skipped = load_samples(
        args.crops, args.manifest, args.include_uncertain
    )
    if args.limit is not None:
        samples = samples[: args.limit]
    fit_samples = None
    if args.fit_crops is not None:
        fit_samples, fit_skipped = load_samples(
            args.fit_crops, args.fit_manifest, args.include_uncertain
        )
        skipped.extend(fit_skipped)

    if args.scale_source == "theory" and any(
        sample.theory is None for sample in [*samples, *(fit_samples or [])]
    ):
        raise ValueError(
            "--scale-source theory needs client and uiPercent in every crop."
        )
    renderer = Renderer(
        FontPaths(
            gulim=args.gulim,
            batang=args.batang,
            nanum=args.nanum,
            uttum=args.uttum,
        )
    )
    settings = {
        "candidates": args.candidates,
        "holdouts": args.holdouts,
        "residuals": args.residuals,
        "iterations": args.iterations,
        "scale_source": args.scale_source,
        "origins": args.origins,
        "captures": args.captures,
        "workers": args.workers,
        "profile": args.profile,
        "color": "auto" if args.color is None else list(args.color),
        "advance": args.advance,
        "crops": str(args.crops),
        "manifest": str(args.manifest),
        "fit_crops": None if args.fit_crops is None else str(args.fit_crops),
        "kernels": None if args.kernels is None else str(args.kernels),
        "scale_model": args.scale_model,
    }
    run_measured_kernel(
        args.out,
        samples,
        skipped,
        lambda sample: _prepare_sample(args, renderer, sample),
        settings,
        fit_samples,
        _scoring_kernels(args),
    )
    print(args.out / "measured-kernel.md")


def model_kernels(model: ScaleModel) -> dict[str, dict[str, FittedKernel]]:
    """Return a kernel model's taps as the ``measured-native`` kernel.

    The taps score every crop the way ``--kernels`` scores a loaded
    ``kernels.json``; the scale only feeds the reported integrals,
    which a native kernel does not depend on.
    """
    if model.kernel is None or model.kernel.spacing != TAP_SPACING["native"]:
        raise ValueError(f"{model.name} has no native-unit kernel.")

    taps = KernelTaps(model.kernel.x, model.kernel.y, "native")

    return {"measured-native": {"all": FittedKernel(taps, 1.0)}}


def _scoring_kernels(
    args: argparse.Namespace,
) -> dict[str, dict[str, FittedKernel]] | None:
    if args.scale_model is not None:
        return model_kernels(SCALE_MODELS[args.scale_model])

    if args.kernels is not None:
        return load_kernels(args.kernels)

    return None


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
        return raid_fill_color(crop)

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
    glyphs = raid_layout(renderer, sample.text, args.profile, args.advance)
    crop = _load_crop(Path(sample.file))
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


RAID_SEPARATOR_LINES = range(1, 12)
LINE_SEARCH_ROWS = 2


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
    """Measure each separator line under the name column per case.

    The heading anchor is recovered from each case's row 1 crop. Every
    line between rows 1 to 12 is measured on its own darkest-channel
    profile, so lines at different sub-pixel phases are not averaged
    into one; the case keeps the median width and the per-line
    widths and centers (rows from the top of the capture).
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
        lines = []
        for line in RAID_SEPARATOR_LINES:
            expected = round(
                anchor_y
                + (RAID_HEADING_HEIGHT + RAID_ROW_SPACING * line) * scale
            )
            # The expected row drifts a row or two from the line on
            # some captures, so center the window on the darkest row
            # nearby before measuring.
            wide = line_profile(
                darkest, x, width, expected, window + LINE_SEARCH_ROWS
            )
            darkest_row = wide.rows[
                min(range(len(wide.values)), key=wide.values.__getitem__)
            ]
            profile = line_profile(darkest, x, width, darkest_row, window)
            lines.append(
                {
                    "line": line,
                    "values": [round(value, 2) for value in profile.values],
                    "width_rows": profile.width_rows,
                    "center_row": profile.rows[0]
                    + line_center(profile.values),
                }
            )
        results.append(
            {
                "caseId": case_id,
                "scale": scale,
                "width_rows": float(
                    np.median([line["width_rows"] for line in lines])
                ),
                "lines": lines,
            }
        )

    return results


def line_width_fit(results: list[dict]) -> dict:
    """Fit ``width = slope * scale + intercept`` over every scaled line.

    Only lines of cases with scale above 1 enter, since s=1 lines are
    exactly two rows and would anchor the fit.
    """
    points = [
        (result["scale"], line["width_rows"])
        for result in results
        if result["scale"] > 1
        for line in result["lines"]
        if not math.isnan(line["width_rows"])
    ]
    if len(points) < 2:
        return {"lines": len(points)}
    scales, widths = np.array(points).T
    slope, intercept = np.polyfit(scales, widths, 1)

    return {
        "lines": len(points),
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(np.corrcoef(scales, widths)[0, 1]),
    }


def run_raid_lines(args: argparse.Namespace) -> None:
    """Print one JSON object per case, then the width-to-scale fit."""
    index = json.loads(args.crops.read_text(encoding="utf-8"))
    results = raid_separator_profiles(index, args.originals, args.window)
    for result in results:
        print(json.dumps(result, ensure_ascii=False))
    print(json.dumps({"fit": line_width_fit(results)}, ensure_ascii=False))


def _parse_phase(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration phase",
        description=(
            "Compare the sub-pixel phase of the text origins with the "
            "separator line centers per case, to see whether captures "
            "at the same scale but another window position move text "
            "and lines together."
        ),
    )
    parser.add_argument(
        "--measured",
        type=Path,
        required=True,
        help="measured-kernel.json of the candidate to read origins from",
    )
    parser.add_argument(
        "--candidate", default="measured-bucket", help="Candidate name"
    )
    parser.add_argument(
        "--mode",
        default="in_sample",
        help="Evaluation whose origins to read, such as holdout_set",
    )
    parser.add_argument(
        "--lines", type=Path, required=True, help="raid-lines JSONL output"
    )
    parser.add_argument(
        "--crops",
        type=Path,
        required=True,
        help="Crop index the origins were fitted on, for the crop offsets",
    )

    return parser.parse_args(argv)


def crop_origins(index: dict) -> dict[str, tuple[int, int]]:
    """Map every crop id of ``index`` to its top-left in the capture."""
    origins = {}
    for position, entry in enumerate(_require(index, "crops", "crop index")):
        where = f"crops[{position}]"
        origin = _crop_origin(entry)
        if origin is None:
            raise ValueError(f"{where} lacks 'crop'.")
        case_id = _require(entry, "caseId", where)
        origins[f"{case_id}_slot{_require(entry, 'slot', where)}"] = origin

    return origins


def phase_table(
    measured: dict,
    candidate: str,
    lines: list[dict],
    crops: dict[str, tuple[int, int]],
    mode: str = "in_sample",
) -> list[dict]:
    """Join each case's text-origin phases with its line-center phases.

    Per row crop the fractional parts of the fitted origin (x and y)
    under ``mode`` are listed, per separator line the fractional part
    of its center. ``row_scale`` and ``line_scale`` divide the span
    between the first and last origin or center by the native row
    spacing, an estimate of the scale that does not use the detected
    one. The fitted origins are crop-relative, so ``crops`` (from
    ``crop_origins``) moves them into the capture before the span is
    taken. Two captures of the same scale whose window moved together
    show the same per-row and per-line phase shift.
    """
    entry = next(
        (item for item in measured["candidates"] if item["name"] == candidate),
        None,
    )
    if entry is None:
        raise ValueError(f"{candidate!r} is not in the measured report.")

    if mode not in entry["evaluations"]:
        raise ValueError(f"{candidate!r} has no {mode!r} evaluation.")
    origins: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    for fit in entry["evaluations"][mode]["fits"]:
        case_id, slot = fit["sample"].rsplit("_slot", 1)
        if fit["sample"] not in crops:
            raise ValueError(f"{fit['sample']} is not in the crop index.")
        crop_x, crop_y = crops[fit["sample"]]
        origins[case_id][int(slot)] = (
            fit["origin"][0] + crop_x,
            fit["origin"][1] + crop_y,
        )
    rows = []
    for result in lines:
        case_id = result["caseId"]
        if case_id not in origins:
            continue
        slots = sorted(origins[case_id])
        centers = {
            line["line"]: line["center_row"]
            for line in result["lines"]
            if not math.isnan(line["center_row"])
        }
        rows.append(
            {
                "caseId": case_id,
                "scale": result["scale"],
                "rows": {
                    str(slot): [
                        round(origins[case_id][slot][0] % 1, 3),
                        round(origins[case_id][slot][1] % 1, 3),
                    ]
                    for slot in slots
                },
                "lines": {
                    str(line): round(center % 1, 3)
                    for line, center in sorted(centers.items())
                },
                "row_scale": _span_scale(
                    {slot: origins[case_id][slot][1] for slot in slots}
                ),
                "line_scale": _span_scale(centers),
            }
        )

    return sorted(rows, key=lambda row: (row["scale"], row["caseId"]))


def _span_scale(positions: dict[int, float]) -> float:
    """Divide the first-to-last span by the native rows it covers."""
    if len(positions) < 2:
        return math.nan
    first, last = min(positions), max(positions)

    return round(
        (positions[last] - positions[first])
        / (RAID_ROW_SPACING * (last - first)),
        4,
    )


def run_phase(args: argparse.Namespace) -> None:
    """Print one JSON object per case with text and line phases."""
    measured = json.loads(args.measured.read_text(encoding="utf-8"))
    lines = [
        json.loads(line)
        for line in args.lines.read_text(encoding="utf-8").splitlines()
        if line.strip() and "caseId" in line
    ]
    crops = crop_origins(json.loads(args.crops.read_text(encoding="utf-8")))
    rows = phase_table(measured, args.candidate, lines, crops, args.mode)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))


def _parse_sharpness(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m dnf_ocr_synth.calibration sharpness",
        description=(
            "Classify every capture of a crop index as sharp or blurred "
            "by the pure-black share of its crops."
        ),
    )
    parser.add_argument("--crops", type=Path, required=True)

    return parser.parse_args(argv)


def capture_sharpness_table(index: dict, crops_dir: Path) -> list[dict]:
    """Return one record per capture with its pure-black share."""
    by_case: dict[str, list[np.ndarray]] = defaultdict(list)
    scales = {}
    for position, entry in enumerate(_require(index, "crops", "crop index")):
        where = f"crops[{position}]"
        case_id = _require(entry, "caseId", where)
        by_case[case_id].append(
            _load_crop(crops_dir / _require(entry, "file", where))
        )
        scales[case_id] = float(_require(entry, "scale", where))
    rows = []
    for case_id, crops in by_case.items():
        ratio = pure_black_ratio(crops)
        rows.append(
            {
                "caseId": case_id,
                "scale": scales[case_id],
                "crops": len(crops),
                "pure_black_ratio": round(ratio, 4),
                "sharp": is_sharp(ratio),
            }
        )

    return rows


def run_sharpness(args: argparse.Namespace) -> None:
    """Print one JSON object per capture with its sharpness verdict."""
    index = json.loads(args.crops.read_text(encoding="utf-8"))
    for row in capture_sharpness_table(index, args.crops.parent):
        print(json.dumps(row, ensure_ascii=False))


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the fit (default) or a measurement subcommand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments[:1] == ["hp-edge"]:
            run_hp_edge(_parse_hp_edge(arguments[1:]))
        elif arguments[:1] == ["raid-lines"]:
            run_raid_lines(_parse_raid_lines(arguments[1:]))
        elif arguments[:1] == ["phase"]:
            run_phase(_parse_phase(arguments[1:]))
        elif arguments[:1] == ["sharpness"]:
            run_sharpness(_parse_sharpness(arguments[1:]))
        elif arguments[:1] == ["measured-kernel"]:
            run_measured(_parse_measured(arguments[1:]))
        else:
            run_fit(_parse_fit(arguments))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
