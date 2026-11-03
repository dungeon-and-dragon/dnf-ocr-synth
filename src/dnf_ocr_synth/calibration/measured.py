"""Compare measured and standard kernels on captured crops.

Every candidate kernel meets the crops under the same conditions: the
same estimated background, the same coverage composite, the same
darkest-channel alignment objective and the same sub-pixel x/y origin
search. Candidates with parameters (measured taps, a brightness gain)
are fitted on plain rows and scored in-sample and under hold-out by
capture or by raid group; fixed shapes have nothing to fit.
"""

import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..render import GlyphLayout
from .align import PLAIN_FILL
from .kernel import (
    STANDARD_SHAPES,
    Kernel,
    KernelSample,
    KernelTaps,
    StandardKernel,
    align_sample,
    estimate_kernel,
    fit_gain,
    model_composite,
    resample,
    tent_taps,
)
from .metrics import Reference, score_composite
from .report import (
    SampleInfo,
    Skipped,
    markdown_table,
    percent,
    residual_sheet,
    scale_bucket,
)

GATE_RATIO = 0.02
ALIGNMENT_PASSES = 2
UNIT_GROUP = "s=1"
HOLDOUTS = ("capture", "raid")
SCALE_SOURCES = ("detected", "theory")


def scale_group(scale: float) -> str:
    """Group exact s=1 crops apart from the rest of the 1.0 bucket."""
    if scale == 1:
        return UNIT_GROUP

    return scale_bucket(scale)


@dataclass(frozen=True)
class Candidate:
    """Describe how one kernel is built and what it fits.

    ``kind`` is ``taps`` (measured taps, ``unit`` crop or native),
    ``gain`` (a standard ``shape`` with a fitted brightness gain) or
    ``fixed`` (a standard ``shape`` as is). ``scope`` says whether
    parameters are fitted per scale group or once on every scaled
    crop (``all``). ``transform`` derives the kernel from the fitted
    taps of ``base`` instead of fitting again.
    """

    name: str
    kind: str
    scope: str = "bucket"
    unit: str | None = None
    shape: str | None = None
    base: str | None = None
    transform: Callable[[KernelTaps, float], Kernel] | None = None

    @property
    def fitted(self) -> bool:
        """Tell whether the candidate has parameters to estimate."""
        return self.kind != "fixed"

    def start(self, scale: float) -> Kernel:
        """Return the kernel before any fitting."""
        if self.kind == "taps":
            return tent_taps(scale, self.unit)

        return StandardKernel(self.shape)

    def fit(
        self, train: list[KernelSample], scale: float, iterations: int
    ) -> Kernel:
        """Fit the parameters on the aligned plain-row ``train``."""
        if self.kind == "taps":
            return estimate_kernel(train, scale, iterations, self.unit)

        if self.kind == "gain":
            return StandardKernel(
                self.shape, fit_gain(train, self.start(scale))
            )

        return self.start(scale)


def _unit_gain(taps: KernelTaps, scale: float) -> Kernel:
    return taps.unit_gain(scale)


CANDIDATES = {
    "measured-bucket": Candidate("measured-bucket", "taps", unit="crop"),
    "measured-bucket-unit": Candidate(
        "measured-bucket-unit",
        "taps",
        unit="crop",
        base="measured-bucket",
        transform=_unit_gain,
    ),
    "measured-native": Candidate(
        "measured-native", "taps", scope="all", unit="native"
    ),
    "measured-native-unit": Candidate(
        "measured-native-unit",
        "taps",
        scope="all",
        unit="native",
        base="measured-native",
        transform=_unit_gain,
    ),
    **{
        shape: Candidate(shape, "fixed", shape=shape)
        for shape in STANDARD_SHAPES
    },
    **{
        f"{shape}-gain": Candidate(f"{shape}-gain", "gain", shape=shape)
        for shape in STANDARD_SHAPES
    },
}
DEFAULT_RESIDUALS = ("measured-bucket", "measured-native")


@dataclass(frozen=True)
class Prepared:
    """Hold what one sample needs before alignment."""

    info: SampleInfo
    glyphs: GlyphLayout
    reference: Reference
    color: tuple[int, int, int]
    scale: float

    @property
    def plain(self) -> bool:
        """Tell whether the row feeds parameter estimation."""
        return self.color == PLAIN_FILL

    @property
    def group(self) -> str:
        """Return the scale group the sample is scored in."""
        return scale_group(self.info.scale)


@dataclass(frozen=True)
class KernelFit:
    """Record one candidate's score on one sample."""

    sample: str
    group: str
    plain: bool
    origin: tuple[float, float]
    mae: float
    mismatch: int
    ink: int
    fold: str = ""

    @property
    def ratio(self) -> float:
        """Return mismatch pixels per crop ink pixel."""
        if self.ink:
            return self.mismatch / self.ink

        return math.inf

    @property
    def passes(self) -> bool:
        """Tell whether the crop alone meets the mismatch gate."""
        return self.ratio <= GATE_RATIO


@dataclass(frozen=True)
class GroupSummary:
    """Aggregate the fits of one scale group."""

    samples: int
    plain_samples: int
    mismatch: int
    ink: int
    plain_mismatch: int
    plain_ink: int
    passed: int
    plain_mae: float

    @property
    def ratio(self) -> float:
        """Return the summed mismatch over the summed ink."""
        if self.ink:
            return self.mismatch / self.ink

        return math.inf

    @property
    def plain_ratio(self) -> float:
        """Return the plain-row mismatch over plain-row ink."""
        if self.plain_ink:
            return self.plain_mismatch / self.plain_ink

        return math.inf

    @property
    def pass_rate(self) -> float:
        """Return the share of crops meeting the gate on their own."""
        if self.samples:
            return self.passed / self.samples

        return math.nan


def summarize(fits: list[KernelFit]) -> GroupSummary:
    """Sum mismatches, count passing crops and average plain MAE."""
    plain = [fit for fit in fits if fit.plain]
    mismatch = sum(fit.mismatch for fit in fits)
    ink = sum(fit.ink for fit in fits)
    plain_mismatch = sum(fit.mismatch for fit in plain)
    plain_ink = sum(fit.ink for fit in plain)
    passed = sum(fit.passes for fit in fits)
    plain_mae = math.nan
    if plain:
        plain_mae = sum(fit.mae for fit in plain) / len(plain)

    return GroupSummary(
        len(fits),
        len(plain),
        mismatch,
        ink,
        plain_mismatch,
        plain_ink,
        passed,
        plain_mae,
    )


@dataclass(frozen=True)
class FittedKernel:
    """Pair a kernel with the mean scale of its fitting crops."""

    kernel: Kernel
    scale: float

    def record(self) -> dict:
        """Describe the kernel as JSON-friendly values."""
        integrals = {
            "integral_x": self.kernel.integral("x", self.scale),
            "integral_y": self.kernel.integral("y", self.scale),
        }
        if isinstance(self.kernel, KernelTaps):
            return {
                "unit": self.kernel.unit,
                "tap_distances": self.kernel.distances,
                "taps_x": list(self.kernel.x),
                "taps_y": list(self.kernel.y),
                **integrals,
            }

        return {
            "shape": self.kernel.shape,
            "gain": self.kernel.gain,
            **integrals,
        }

    def describe(self) -> str:
        """Describe the kernel for the Markdown report."""
        kernel = self.kernel
        if isinstance(kernel, KernelTaps):
            return (
                f"{kernel.unit} 단위, 적분 x "
                f"{kernel.integral('x', self.scale):.3f}, y "
                f"{kernel.integral('y', self.scale):.3f}; x: "
                + ", ".join(f"{value:.3f}" for value in kernel.x)
                + "; y: "
                + ", ".join(f"{value:.3f}" for value in kernel.y)
            )

        return f"{kernel.shape}, gain {kernel.gain:.4f}"


@dataclass
class Evaluation:
    """Hold one candidate's fits under one evaluation mode."""

    mode: str
    fits: list[KernelFit]
    kernels: dict[str, FittedKernel]
    unevaluated: list[str]

    def by_group(self) -> dict[str, GroupSummary]:
        """Summarize the fits per scale group."""
        grouped: dict[str, list[KernelFit]] = defaultdict(list)
        for fit in self.fits:
            grouped[fit.group].append(fit)

        return {
            group: summarize(grouped[group])
            for group in sorted(grouped, key=group_order)
        }


def group_title(group: str) -> str:
    """Name a scale group for the report: ``s=1`` or ``s 1.8 구간``."""
    if group == UNIT_GROUP:
        return f"{group} 구간"

    return f"s {group} 구간"


def group_order(group: str) -> tuple[int, str]:
    """Sort the s=1 group before the scale buckets."""
    rank = 1
    if group == UNIT_GROUP:
        rank = 0

    return (rank, group)


@dataclass
class CandidateResult:
    """Collect a candidate's evaluations and the composites to draw."""

    candidate: Candidate
    evaluations: dict[str, Evaluation]
    composites: dict[str, np.ndarray]


def _align_one(item: Prepared, kernel: Kernel | None) -> KernelSample:
    return align_sample(
        item.reference,
        item.glyphs,
        float(min(item.color)),
        item.scale,
        kernel,
    )


class Aligner:
    """Align samples with a kernel, optionally on worker processes."""

    def __init__(self, workers: int) -> None:
        """Keep ``workers`` processes; 1 aligns in this process."""
        self._pool = ProcessPoolExecutor(workers) if workers > 1 else None

    def close(self) -> None:
        """Shut the worker processes down."""
        if self._pool is not None:
            self._pool.shutdown()

    def align(
        self, prepared: list[Prepared], kernel: Kernel | None
    ) -> list[KernelSample]:
        """Return one aligned sample per prepared crop, in order."""
        if self._pool is None:
            return [_align_one(item, kernel) for item in prepared]

        return list(
            self._pool.map(
                _align_one, prepared, [kernel] * len(prepared), chunksize=4
            )
        )


def score_sample(
    item: Prepared, sample: KernelSample, kernel: Kernel, fold: str = ""
) -> tuple[KernelFit, np.ndarray]:
    """Composite the aligned sample and score it against the crop."""
    coverage, filled = resample(
        sample.fill,
        sample.alpha,
        sample.scale,
        sample.origin,
        kernel,
        item.reference.ink.shape,
    )
    composite, covered = model_composite(
        item.reference, coverage, filled, item.color
    )
    comparison = score_composite(item.reference, composite, covered)
    fit = KernelFit(
        item.info.id,
        item.group,
        item.plain,
        sample.origin,
        comparison.mae,
        comparison.mismatch,
        comparison.ink,
        fold,
    )

    return fit, composite


def _pools(
    candidate: Candidate, prepared: list[Prepared]
) -> dict[str, list[Prepared]]:
    """Split the samples into the sets the candidate is fitted on.

    Fitted candidates leave the s=1 crops out: at s=1 the native
    raster is copied as is, so there is no kernel to measure there.
    """
    if candidate.fitted:
        prepared = [item for item in prepared if item.group != UNIT_GROUP]

    if candidate.scope == "all":
        if prepared:
            return {"all": prepared}

        return {}
    pools: dict[str, list[Prepared]] = defaultdict(list)
    for item in prepared:
        pools[item.group].append(item)

    return {group: pools[group] for group in sorted(pools, key=group_order)}


def _pool_scale(pool: list[Prepared]) -> float:
    return sum(item.scale for item in pool) / len(pool)


def fit_in_sample(
    candidate: Candidate,
    pool: list[Prepared],
    iterations: int,
    aligner: Aligner,
) -> tuple[Kernel, list[KernelSample]]:
    """Fit the candidate on the pool's plain rows and align everything.

    Alignment starts from the unfitted kernel, the parameters are
    estimated from the plain rows, and both repeat so the placement
    matches the fitted kernel's edges. Returns the final kernel and the
    final alignment of every sample.
    """
    scale = _pool_scale(pool)
    kernel = candidate.start(scale)
    passes = ALIGNMENT_PASSES if candidate.fitted else 0
    for _ in range(passes):
        samples = aligner.align(pool, kernel)
        train = [sample for sample, item in zip(samples, pool) if item.plain]
        kernel = candidate.fit(train, scale, iterations)

    return kernel, aligner.align(pool, kernel)


def _holdout_key(item: Prepared, holdout: str) -> str:
    if holdout == "capture":
        return item.info.case_id

    return item.info.raid


def _folds(pool: list[Prepared], holdout: str) -> dict[str, list[int]]:
    folds: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(pool):
        folds[_holdout_key(item, holdout)].append(index)

    return dict(sorted(folds.items()))


def evaluate_holdout(
    candidate: Candidate,
    pool: list[Prepared],
    samples: list[KernelSample],
    holdout: str,
    iterations: int,
    aligner: Aligner,
) -> tuple[list[KernelFit], dict[str, FittedKernel], list[str]]:
    """Score every sample with a kernel fitted without its fold.

    The training rows keep the origins of the in-sample alignment;
    only the held-out crops are re-aligned with the fold's kernel.
    Folds without a plain training row are left unevaluated.
    """
    scale = _pool_scale(pool)
    fits = []
    kernels = {}
    unevaluated = []
    for fold, held in _folds(pool, holdout).items():
        train = [
            sample
            for index, (sample, item) in enumerate(zip(samples, pool))
            if item.plain and index not in held
        ]
        if not train:
            unevaluated.extend(pool[index].info.id for index in held)
            continue
        kernel = candidate.fit(train, scale, iterations)
        kernels[fold] = FittedKernel(kernel, scale)
        aligned = aligner.align([pool[index] for index in held], kernel)
        for index, sample in zip(held, aligned):
            fits.append(score_sample(pool[index], sample, kernel, fold)[0])

    return fits, kernels, unevaluated


def derived_holdout(
    candidate: Candidate,
    pool: list[Prepared],
    holdout: str,
    base_kernels: dict[str, FittedKernel],
    aligner: Aligner,
) -> tuple[list[KernelFit], dict[str, FittedKernel], list[str]]:
    """Score every fold with the transformed kernel of the base fold."""
    fits = []
    kernels = {}
    unevaluated = []
    for fold, held in _folds(pool, holdout).items():
        fitted = base_kernels.get(fold)
        if fitted is None:
            unevaluated.extend(pool[index].info.id for index in held)
            continue
        kernel = candidate.transform(fitted.kernel, fitted.scale)
        kernels[fold] = FittedKernel(kernel, fitted.scale)
        aligned = aligner.align([pool[index] for index in held], kernel)
        for index, sample in zip(held, aligned):
            fits.append(score_sample(pool[index], sample, kernel, fold)[0])

    return fits, kernels, unevaluated


def _pool_kernels(
    kernels: dict[str, FittedKernel], pool_name: str
) -> dict[str, FittedKernel]:
    prefix = f"{pool_name}/"

    return {
        key[len(prefix) :]: fitted
        for key, fitted in kernels.items()
        if key.startswith(prefix)
    }


def evaluate_candidate(
    candidate: Candidate,
    prepared: list[Prepared],
    holdouts: tuple[str, ...],
    iterations: int,
    aligner: Aligner,
    base: CandidateResult | None = None,
) -> CandidateResult:
    """Run the in-sample fit and every hold-out for one candidate.

    A derived candidate reuses the fitted kernels of ``base`` through
    its ``transform`` instead of fitting again.
    """
    if candidate.transform is not None and base is None:
        raise ValueError(
            f"{candidate.name} needs the {candidate.base} result first."
        )

    in_sample = Evaluation("in_sample", [], {}, [])
    holdout_evaluations = {
        holdout: Evaluation(f"holdout_{holdout}", [], {}, [])
        for holdout in holdouts
        if candidate.fitted
    }
    composites = {}
    for pool_name, pool in _pools(candidate, prepared).items():
        if candidate.fitted and not any(item.plain for item in pool):
            in_sample.unevaluated.extend(item.info.id for item in pool)
            continue

        if candidate.transform is not None:
            fitted = base.evaluations["in_sample"].kernels[pool_name]
            kernel = candidate.transform(fitted.kernel, fitted.scale)
            samples = aligner.align(pool, kernel)
        else:
            kernel, samples = fit_in_sample(
                candidate, pool, iterations, aligner
            )
        in_sample.kernels[pool_name] = FittedKernel(kernel, _pool_scale(pool))
        for item, sample in zip(pool, samples):
            fit, composite = score_sample(item, sample, kernel)
            in_sample.fits.append(fit)
            composites[item.info.id] = composite
        for holdout, evaluation in holdout_evaluations.items():
            if candidate.transform is not None:
                fits, kernels, unevaluated = derived_holdout(
                    candidate,
                    pool,
                    holdout,
                    _pool_kernels(
                        base.evaluations[evaluation.mode].kernels, pool_name
                    ),
                    aligner,
                )
            else:
                fits, kernels, unevaluated = evaluate_holdout(
                    candidate, pool, samples, holdout, iterations, aligner
                )
            evaluation.fits.extend(fits)
            evaluation.kernels.update(
                {
                    f"{pool_name}/{fold}": fitted
                    for fold, fitted in kernels.items()
                }
            )
            evaluation.unevaluated.extend(unevaluated)

    evaluations = {"in_sample": in_sample}
    for evaluation in holdout_evaluations.values():
        evaluations[evaluation.mode] = evaluation

    return CandidateResult(candidate, evaluations, composites)


def score_with_kernels(
    candidate: Candidate,
    prepared: list[Prepared],
    kernels: dict[str, FittedKernel],
    aligner: Aligner,
) -> tuple[Evaluation, dict[str, np.ndarray]]:
    """Score separate crops with kernels fitted on another set."""
    evaluation = Evaluation("holdout_set", [], dict(kernels), [])
    composites = {}
    for pool_name, pool in _pools(candidate, prepared).items():
        fitted = kernels.get(pool_name)
        if fitted is None:
            evaluation.unevaluated.extend(item.info.id for item in pool)
            continue
        for item, sample in zip(pool, aligner.align(pool, fitted.kernel)):
            fit, composite = score_sample(
                item, sample, fitted.kernel, pool_name
            )
            evaluation.fits.append(fit)
            composites[item.info.id] = composite

    return evaluation, composites


@dataclass(frozen=True)
class RowKind:
    """Record how one raid row was classified across its crops."""

    raid: str
    slot: int
    text: str
    plain: int
    glow: int


def row_kinds(prepared: list[Prepared]) -> list[RowKind]:
    """List every raid row with its plain and glow crop counts."""
    counts: dict[tuple[str, int, str], list[int]] = defaultdict(lambda: [0, 0])
    for item in prepared:
        key = (item.info.raid, item.info.slot, item.info.text)
        counts[key][0 if item.plain else 1] += 1

    return [
        RowKind(raid, slot, text, plain, glow)
        for (raid, slot, text), (plain, glow) in sorted(counts.items())
    ]


def prepare_all(
    samples: list[SampleInfo],
    skipped: list[Skipped],
    prepare: Callable[
        [SampleInfo], tuple[GlyphLayout, Reference, tuple[int, int, int]]
    ],
    scale_source: str,
) -> list[Prepared]:
    """Render and load every sample, skipping the ones that fail."""
    if scale_source not in SCALE_SOURCES:
        raise ValueError(f"scale source must be one of {SCALE_SOURCES}.")

    prepared = []
    for sample in samples:
        try:
            glyphs, reference, color = prepare(sample)
        except (ValueError, FileNotFoundError) as error:
            skipped.append(Skipped(sample.id, f"렌더 실패: {error}"))
            continue
        scale = sample.scale if scale_source == "detected" else sample.theory
        prepared.append(Prepared(sample, glyphs, reference, color, scale))

    return prepared


def run_measured_kernel(
    out_dir: Path,
    samples: list[SampleInfo],
    skipped: list[Skipped],
    prepare: Callable[
        [SampleInfo], tuple[GlyphLayout, Reference, tuple[int, int, int]]
    ],
    settings: dict,
    fit_samples: list[SampleInfo] | None = None,
) -> None:
    """Evaluate the candidates and write the report and residuals.

    ``prepare(sample)`` returns ``(glyphs, reference, color)`` or
    raises ValueError or FileNotFoundError for a sample to skip.
    ``settings`` carries ``candidates``, ``holdouts``, ``residuals``,
    ``iterations``, ``scale_source`` and ``workers``. With
    ``fit_samples`` the parameters come from those crops, whose
    in-sample scores stay in the report, and every ``samples`` crop is
    scored as ``holdout_set``.
    """
    started = time.monotonic()
    prepared = prepare_all(samples, skipped, prepare, settings["scale_source"])
    fitted = (
        prepared
        if fit_samples is None
        else prepare_all(
            fit_samples, skipped, prepare, settings["scale_source"]
        )
    )
    aligner = Aligner(settings["workers"])
    results: dict[str, CandidateResult] = {}
    try:
        for name in settings["candidates"]:
            candidate = CANDIDATES[name]
            result = evaluate_candidate(
                candidate,
                fitted,
                tuple(settings["holdouts"]) if fit_samples is None else (),
                settings["iterations"],
                aligner,
                results.get(candidate.base),
            )
            if fit_samples is not None:
                evaluation, composites = score_with_kernels(
                    candidate,
                    prepared,
                    result.evaluations["in_sample"].kernels,
                    aligner,
                )
                result.evaluations[evaluation.mode] = evaluation
                result.composites = composites
            results[name] = result
            _print_progress(name, result, started)
    finally:
        aligner.close()
    residual_dir = out_dir / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    by_id = {item.info.id: item for item in prepared}
    for name in settings["residuals"]:
        for sample_id, composite in results[name].composites.items():
            residual_sheet(by_id[sample_id].reference, composite).save(
                residual_dir / f"{sample_id}_{name}.png"
            )
    write_measured_report(
        out_dir, list(results.values()), row_kinds(prepared), skipped, settings
    )


def _print_progress(
    name: str, result: CandidateResult, started: float
) -> None:
    for evaluation in result.evaluations.values():
        summary = summarize(evaluation.fits)
        print(
            f"[{name}] {evaluation.mode}: {summary.samples} samples, "
            f"mismatch {summary.mismatch}/{summary.ink} "
            f"({percent(summary.ratio)}), pass {summary.passed} "
            f"{time.monotonic() - started:.0f}s",
            file=sys.stderr,
        )


def _summary_record(summary: GroupSummary) -> dict:
    return {
        **asdict(summary),
        "ratio": summary.ratio,
        "plain_ratio": summary.plain_ratio,
        "pass_rate": summary.pass_rate,
    }


def write_measured_report(
    out_dir: Path,
    results: list[CandidateResult],
    rows: list[RowKind],
    skipped: list[Skipped],
    settings: dict,
) -> None:
    """Write ``measured-kernel.json`` and ``measured-kernel.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "dnf_ocr_synth.calibration measured-kernel",
        "settings": settings,
        "gate_ratio": GATE_RATIO,
        "rows": [asdict(row) for row in rows],
        "candidates": [
            {
                "name": result.candidate.name,
                "kind": result.candidate.kind,
                "scope": result.candidate.scope,
                "evaluations": {
                    evaluation.mode: {
                        "groups": {
                            group: _summary_record(summary)
                            for group, summary in evaluation.by_group().items()
                        },
                        "kernels": {
                            key: fitted.record()
                            for key, fitted in evaluation.kernels.items()
                        },
                        "unevaluated": evaluation.unevaluated,
                        "fits": [
                            {**asdict(fit), "ratio": fit.ratio}
                            for fit in evaluation.fits
                        ],
                    }
                    for evaluation in result.evaluations.values()
                },
            }
            for result in results
        ],
        "skipped": [asdict(entry) for entry in skipped],
    }
    (out_dir / "measured-kernel.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "measured-kernel.md").write_text(
        _markdown(results, rows, skipped, settings), encoding="utf-8"
    )


def _mode_columns(summary: GroupSummary | None) -> list[str]:
    if summary is None:
        return ["-", "-", "-"]

    return [
        f"{summary.mismatch}/{summary.ink} ({percent(summary.ratio)})",
        f"{summary.passed}/{summary.samples}",
        f"{summary.plain_mae:.1f}",
    ]


def _markdown(
    results: list[CandidateResult],
    rows: list[RowKind],
    skipped: list[Skipped],
    settings: dict,
) -> str:
    modes = []
    for result in results:
        for mode in result.evaluations:
            if mode not in modes:
                modes.append(mode)
    groups = sorted(
        {
            fit.group
            for result in results
            for evaluation in result.evaluations.values()
            for fit in evaluation.fits
        },
        key=group_order,
    )
    sections = [
        "# 측정 커널과 표준 필터 비교",
        "",
        "- 모든 후보가 같은 배경 추정, 같은 coverage 합성, 같은 가장 어두운 "
        "채널 정렬 목적함수, 같은 x/y 서브픽셀 원점 탐색으로 채점됐다.",
        "- 측정 커널(measured-*)은 음수 tap과 1을 넘는 coverage를 허용하고 "
        "넓이도 제약하지 않는다. `-unit`은 그 tap을 x, y 넓이 곱이 1이 "
        "되게 나눈 것이다. 표준 필터는 음수가 없고 넓이 1이며, `-gain`은 "
        "밝기 gain 한 값을 일반 행에서 최소제곱으로 맞춘 것이다.",
        "- 불일치 비율은 불일치 픽셀 합을 실기 잉크 픽셀 합으로 나눈 값, "
        f"통과는 crop 단독 비율이 {percent(GATE_RATIO)} 이하인 crop 수, "
        "MAE는 일반 행만 평균한 값이다.",
        "- in_sample은 채점 crop이 추정에도 쓰인 결과, holdout_capture는 "
        "같은 캡처를 빼고 추정한 결과, holdout_raid는 같은 공대를 빼고 "
        "추정한 결과다. 추정 표본의 원점은 in_sample 정렬을 그대로 쓰고 "
        "채점 crop만 다시 정렬한다. 추정 표본이 없는 fold는 빠진다.",
        f"- 추정 반복 {settings['iterations']}회, 배율 출처 "
        f"{settings['scale_source']}.",
    ]
    for group in groups:
        sections.extend(["", f"## {group_title(group)}", ""])
        headers = ["후보"]
        for mode in modes:
            headers.extend([f"{mode} 불일치", f"{mode} 통과", f"{mode} MAE"])
        table_rows = []
        for result in results:
            row = [result.candidate.name]
            for mode in modes:
                evaluation = result.evaluations.get(mode)
                summary = (
                    evaluation.by_group().get(group) if evaluation else None
                )
                row.extend(_mode_columns(summary))
            table_rows.append(row)
        sections.append(markdown_table(headers, table_rows))
    sections.extend(["", "## 추정한 커널", ""])
    for result in results:
        for evaluation in result.evaluations.values():
            if not evaluation.mode.startswith(("in_sample", "holdout_set")):
                continue
            for key, fitted in evaluation.kernels.items():
                sections.append(
                    f"- {result.candidate.name} [{key}]: {fitted.describe()}"
                )
    sections.extend(["", "## 행 분류", ""])
    sections.append(
        markdown_table(
            ["공대", "행", "이름", "일반 crop", "글로우 crop"],
            [
                [
                    row.raid,
                    str(row.slot),
                    row.text,
                    str(row.plain),
                    str(row.glow),
                ]
                for row in rows
            ],
        )
    )
    for result in results:
        if result.candidate.name not in settings["residuals"]:
            continue
        for evaluation in result.evaluations.values():
            sections.extend(
                [
                    "",
                    f"## {result.candidate.name} {evaluation.mode}: 샘플별",
                    "",
                ]
            )
            sections.append(
                markdown_table(
                    [
                        "샘플",
                        "구간",
                        "일반 행",
                        "fold",
                        "원점",
                        "불일치 (비율)",
                        "MAE",
                    ],
                    [
                        [
                            fit.sample,
                            fit.group,
                            "예" if fit.plain else "아니오",
                            fit.fold,
                            f"({fit.origin[0]:.2f}, {fit.origin[1]:.2f})",
                            f"{fit.mismatch} ({percent(fit.ratio)})",
                            f"{fit.mae:.2f}",
                        ]
                        for fit in sorted(
                            evaluation.fits, key=lambda fit: fit.sample
                        )
                    ],
                )
            )
            if evaluation.unevaluated:
                sections.append("")
                sections.append(
                    "추정 표본이 없어 빠진 crop: "
                    + ", ".join(evaluation.unevaluated)
                )
    if skipped:
        sections.extend(["", "## 건너뛴 샘플", ""])
        sections.append(
            markdown_table(
                ["샘플", "이유"],
                [[entry.sample, entry.reason] for entry in skipped],
            )
        )

    return "\n".join(sections) + "\n"
