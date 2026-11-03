"""Fit one measured kernel per scale bucket and score every crop."""

import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..render import GlyphLayout
from .align import RAID_FILL_COLORS
from .kernel import (
    KernelSample,
    KernelTaps,
    align_sample,
    estimate_kernel,
    model_composite,
    resample,
)
from .metrics import Reference, score_composite
from .report import (
    SampleInfo,
    Skipped,
    markdown_table,
    percent,
    residual_sheet,
)

PIPELINE_NAME = "measured-kernel"
# Rows of the viewer's own party draw the name translucently over a
# glow, so only plain rows feed the kernel estimate; every row is
# still scored.
ESTIMATION_FILL = RAID_FILL_COLORS["beige"]


@dataclass(frozen=True)
class KernelFit:
    """Record the measured-kernel score of one sample."""

    sample: str
    origin: tuple[float, float]
    estimated_from: bool
    mae: float
    mismatch: int
    ink: int

    @property
    def ratio(self) -> float:
        """Return mismatch pixels per crop ink pixel."""
        if self.ink:
            return self.mismatch / self.ink

        return math.inf


@dataclass(frozen=True)
class BucketResult:
    """Summarize one bucket's kernel and scores."""

    bucket: str
    scale_mean: float
    taps: KernelTaps
    integral_x: float
    integral_y: float
    fits: list[KernelFit]

    @property
    def mismatch(self) -> int:
        """Sum mismatches over the bucket."""
        return sum(fit.mismatch for fit in self.fits)

    @property
    def ink(self) -> int:
        """Sum crop ink over the bucket."""
        return sum(fit.ink for fit in self.fits)

    @property
    def mean_ratio(self) -> float:
        """Average the per-sample mismatch ratio."""
        return sum(fit.ratio for fit in self.fits) / len(self.fits)

    @property
    def mean_mae(self) -> float:
        """Average the per-sample MAE."""
        return sum(fit.mae for fit in self.fits) / len(self.fits)


@dataclass(frozen=True)
class Prepared:
    """Hold what one sample needs before alignment."""

    info: SampleInfo
    glyphs: GlyphLayout
    reference: Reference
    color: tuple[int, int, int]


def _align_all(
    prepared: list[Prepared], taps: KernelTaps | None
) -> list[KernelSample]:
    return [
        align_sample(
            item.reference,
            item.glyphs,
            float(min(item.color)),
            item.info.scale,
            taps,
        )
        for item in prepared
    ]


def fit_bucket(
    bucket: str, prepared: list[Prepared], iterations: int
) -> tuple[BucketResult, list[np.ndarray]]:
    """Estimate the bucket kernel and score every prepared sample.

    Alignment runs with the bilinear start, the kernel is estimated
    from the plain-row samples, and both are repeated once with the
    estimated kernel so the placement matches its sharper edges.
    Returns the result and each sample's crop-sized composite.
    """
    scale = sum(item.info.scale for item in prepared) / len(prepared)
    estimation_mask = [item.color == ESTIMATION_FILL for item in prepared]
    taps = None
    for _ in range(2):
        samples = _align_all(prepared, taps)
        taps = estimate_kernel(
            [sample for sample, use in zip(samples, estimation_mask) if use],
            scale,
            iterations,
        )
    samples = _align_all(prepared, taps)
    fits = []
    composites = []
    for item, sample, used in zip(prepared, samples, estimation_mask):
        coverage, filled = resample(
            sample.fill,
            sample.alpha,
            sample.scale,
            sample.origin,
            taps,
            item.reference.ink.shape,
        )
        composite, covered = model_composite(
            item.reference, coverage, filled, item.color
        )
        comparison = score_composite(item.reference, composite, covered)
        fits.append(
            KernelFit(
                item.info.id,
                sample.origin,
                used,
                comparison.mae,
                comparison.mismatch,
                comparison.ink,
            )
        )
        composites.append(composite)
    result = BucketResult(
        bucket,
        scale,
        taps,
        taps.integral("x", scale),
        taps.integral("y", scale),
        fits,
    )

    return result, composites


def run_measured_kernel(
    out_dir: Path,
    samples: list[SampleInfo],
    skipped: list[Skipped],
    prepare: Callable[
        [SampleInfo], tuple[GlyphLayout, Reference, tuple[int, int, int]]
    ],
    iterations: int,
    settings: dict,
) -> None:
    """Fit every bucket, then write the report and residual sheets.

    ``prepare(sample)`` returns ``(glyphs, reference, color)`` or
    raises ValueError or FileNotFoundError for a sample to skip.
    """
    by_bucket: dict[str, list[Prepared]] = defaultdict(list)
    started = time.monotonic()
    for sample in samples:
        try:
            glyphs, reference, color = prepare(sample)
        except (ValueError, FileNotFoundError) as error:
            skipped.append(Skipped(sample.id, f"렌더 실패: {error}"))
            continue
        by_bucket[sample.bucket].append(
            Prepared(sample, glyphs, reference, color)
        )
    residual_dir = out_dir / "residual"
    residual_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for bucket in sorted(by_bucket):
        prepared = by_bucket[bucket]
        if not any(item.color == ESTIMATION_FILL for item in prepared):
            skipped.extend(
                Skipped(item.info.id, f"s {bucket} 구간에 일반 행 crop이 없음")
                for item in prepared
            )
            continue

        result, composites = fit_bucket(bucket, prepared, iterations)
        results.append(result)
        for item, composite in zip(prepared, composites):
            residual_sheet(item.reference, composite).save(
                residual_dir / f"{item.info.id}_{PIPELINE_NAME}.png"
            )
        print(
            f"[{bucket}] {len(prepared)} samples, mismatch "
            f"{result.mismatch}/{result.ink} "
            f"({percent(result.mismatch / max(result.ink, 1))}) "
            f"{time.monotonic() - started:.0f}s",
            file=sys.stderr,
        )
    write_measured_report(out_dir, results, skipped, settings)


def write_measured_report(
    out_dir: Path,
    results: list[BucketResult],
    skipped: list[Skipped],
    settings: dict,
) -> None:
    """Write ``measured-kernel.json`` and ``measured-kernel.md``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "dnf_ocr_synth.calibration measured-kernel",
        "settings": settings,
        "buckets": [
            {
                "bucket": result.bucket,
                "scale_mean": result.scale_mean,
                "tap_distances": result.taps.distances,
                "taps_x": list(result.taps.x),
                "taps_y": list(result.taps.y),
                "integral_x": result.integral_x,
                "integral_y": result.integral_y,
                "samples": len(result.fits),
                "estimation_samples": sum(
                    fit.estimated_from for fit in result.fits
                ),
                "mismatch": result.mismatch,
                "ink": result.ink,
                "mean_ratio": result.mean_ratio,
                "mean_mae": result.mean_mae,
                "fits": [
                    {**asdict(fit), "ratio": fit.ratio} for fit in result.fits
                ],
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
        _markdown(results, skipped, settings), encoding="utf-8"
    )


def _markdown(
    results: list[BucketResult], skipped: list[Skipped], settings: dict
) -> str:
    sections = [
        "# 측정 커널 결과",
        "",
        "- 구간마다 일반 행(beige) crop으로 분리 가능한 대칭 커널(crop px "
        f"0.25 간격 탭)을 최소제곱으로 추정하고, 구간의 모든 crop에 "
        f"적용해 Otsu 불일치와 MAE를 쟀다. 반복 {settings['iterations']}회.",
        "- 불일치 비율은 불일치 픽셀 수를 실기 crop의 잉크 픽셀 수로 "
        "나눈 값이고, 평균 불일치 비율은 샘플별 비율의 평균이다.",
        "- 적분은 커널 넓이를 원본 픽셀 간격으로 나눈 값이다. 1이면 "
        "밝기를 보존한다.",
        "",
        "## 구간별 결과",
        "",
        markdown_table(
            [
                "s 구간",
                "샘플",
                "추정 샘플",
                "불일치/잉크",
                "불일치 비율",
                "평균 불일치 비율",
                "평균 MAE",
                "적분 x, y",
            ],
            [
                [
                    result.bucket,
                    str(len(result.fits)),
                    str(sum(fit.estimated_from for fit in result.fits)),
                    f"{result.mismatch}/{result.ink}",
                    percent(result.mismatch / max(result.ink, 1)),
                    percent(result.mean_ratio),
                    f"{result.mean_mae:.2f}",
                    f"{result.integral_x:.3f}, {result.integral_y:.3f}",
                ]
                for result in results
            ],
        ),
        "",
        "## 구간별 커널 탭 (crop px 거리 0, 0.25, 0.5, ...)",
        "",
    ]
    for result in results:
        sections.append(
            f"- s {result.bucket} (평균 s {result.scale_mean:.3f}) x: "
            + ", ".join(f"{value:.3f}" for value in result.taps.x)
        )
        sections.append(
            f"- s {result.bucket} y: "
            + ", ".join(f"{value:.3f}" for value in result.taps.y)
        )
    for result in results:
        sections.extend(["", f"## s {result.bucket} 구간: 샘플별", ""])
        sections.append(
            markdown_table(
                ["샘플", "추정에 사용", "원점", "불일치 (비율)", "MAE"],
                [
                    [
                        fit.sample,
                        "예" if fit.estimated_from else "아니오",
                        f"({fit.origin[0]:.2f}, {fit.origin[1]:.2f})",
                        f"{fit.mismatch} ({percent(fit.ratio)})",
                        f"{fit.mae:.2f}",
                    ]
                    for fit in result.fits
                ],
            )
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
