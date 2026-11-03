"""Aggregate fits per scale bucket and write report files."""

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .metrics import Reference, bt601_gray, composite_over

RESIDUAL_ZOOM = 4
OBJECTIVE = (
    "오프셋은 Otsu 뒤 불일치 픽셀 수, 그 다음 L MAE가 가장 작은 곳을 고르고, "
    "파이프라인 순위는 s 구간 안 샘플의 평균 불일치 비율, 평균 MAE 순이다."
)


@dataclass(frozen=True)
class SampleInfo:
    """Identify one crop with its label and detected scale."""

    id: str
    case_id: str
    screen: str
    slot: int
    file: str
    scale: float
    text: str
    ui_percent: int | None
    capture: str

    @property
    def bucket(self) -> str:
        """Group scales by 0.1 like the dfragon evaluation script."""
        return scale_bucket(self.scale)


@dataclass(frozen=True)
class Fit:
    """Record the best placement of one pipeline on one sample."""

    sample: str
    pipeline: str
    offset: tuple[int, int]
    origin: tuple[int, int]
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
class Skipped:
    """Explain why a crop was not fitted."""

    sample: str
    reason: str


@dataclass(frozen=True)
class Ranking:
    """Summarize one pipeline over the samples of a bucket."""

    pipeline: str
    samples: int
    mean_ratio: float
    mean_mae: float
    mismatch: int
    ink: int


def scale_bucket(scale: float) -> str:
    """Return the 0.1-wide bucket label, so 1.28 falls in ``1.2``."""
    return f"{math.floor(round(scale * 10, 6)) / 10:.1f}"


def rank_pipelines(
    samples: list[SampleInfo], fits: list[Fit], top: int
) -> dict[str, list[Ranking]]:
    """Rank pipelines per bucket and overall (``all``)."""
    buckets = {sample.id: sample.bucket for sample in samples}
    grouped: dict[str, dict[str, list[Fit]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for fit in fits:
        grouped[buckets[fit.sample]][fit.pipeline].append(fit)
        grouped["all"][fit.pipeline].append(fit)
    rankings = {}
    for bucket in sorted(grouped, key=lambda key: (key == "all", key)):
        entries = [
            Ranking(
                pipeline,
                len(group),
                sum(fit.ratio for fit in group) / len(group),
                sum(fit.mae for fit in group) / len(group),
                sum(fit.mismatch for fit in group),
                sum(fit.ink for fit in group),
            )
            for pipeline, group in grouped[bucket].items()
        ]
        entries.sort(
            key=lambda entry: (
                entry.mean_ratio,
                entry.mean_mae,
                entry.pipeline,
            )
        )
        rankings[bucket] = entries[:top]

    return rankings


def residual_image(
    reference: Reference, synthetic: np.ndarray, origin: tuple[int, int]
) -> Image.Image:
    """Place the synthetic, then stack crop, composite, difference."""
    composite, _ = composite_over(reference, synthetic, origin)

    return residual_sheet(reference, composite)


def residual_sheet(reference: Reference, composite: np.ndarray) -> Image.Image:
    """Stack the crop, a crop-sized composite, and their difference.

    The difference panel shows the gray difference and tints pixels
    red where the product's Otsu result disagrees.
    """
    crop = reference.rgba
    crop_gray = bt601_gray(crop).astype(np.int16)
    difference = np.abs(crop_gray - bt601_gray(composite).astype(np.int16))
    mismatch = (reference.binarize(composite)[..., 0] == 0) != reference.ink
    diff_rgb = np.repeat(difference.astype(np.uint8)[..., None], 3, axis=2)
    diff_rgb[mismatch] = [255, 0, 0]
    panels = [
        Image.fromarray(crop[..., :3], "RGB"),
        Image.fromarray(composite[..., :3], "RGB"),
        Image.fromarray(diff_rgb, "RGB"),
    ]
    width = panels[0].width * RESIDUAL_ZOOM
    height = panels[0].height * RESIDUAL_ZOOM
    sheet = Image.new("RGB", (width, height * 3 + 2), (128, 128, 128))
    for index, panel in enumerate(panels):
        zoomed = panel.resize((width, height), Image.Resampling.NEAREST)
        sheet.paste(zoomed, (0, index * (height + 1)))

    return sheet


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)

    return "\n".join(lines)


def percent(value: float) -> str:
    """Format a ratio as a percentage, or ``inf``."""
    if math.isinf(value):
        return "inf"

    return f"{value * 100:.2f}%"


def write_report(
    out_dir: Path,
    samples: list[SampleInfo],
    fits: list[Fit],
    skipped: list[Skipped],
    rankings: dict[str, list[Ranking]],
    settings: dict,
) -> None:
    """Write ``report.json`` and ``report.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "dnf_ocr_synth.calibration",
        "objective": OBJECTIVE,
        "settings": settings,
        "samples": [asdict(sample) for sample in samples],
        "skipped": [asdict(entry) for entry in skipped],
        "fits": [{**asdict(fit), "ratio": fit.ratio} for fit in fits],
        "rankings": {
            bucket: [asdict(entry) for entry in entries]
            for bucket, entries in rankings.items()
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report.md").write_text(
        _markdown(samples, fits, skipped, rankings, settings),
        encoding="utf-8",
    )


def _markdown(
    samples: list[SampleInfo],
    fits: list[Fit],
    skipped: list[Skipped],
    rankings: dict[str, list[Ranking]],
    settings: dict,
) -> str:
    by_sample: dict[str, list[Fit]] = defaultdict(list)
    for fit in fits:
        by_sample[fit.sample].append(fit)
    radius = settings["radius"]
    at_boundary = sum(
        max(abs(fit.offset[0]), abs(fit.offset[1])) == radius for fit in fits
    )
    outline = (
        "외곽선 없음"
        if settings.get("outline_px") == 0
        else "파이프라인 외곽선"
    )
    sections = [
        "# 글자 확대 가설 맞추기 결과",
        "",
        f"- 파이프라인 {settings['pipeline_count']}개, "
        f"오프셋 반경 {radius}px, 샘플 {len(samples)}개, "
        f"건너뜀 {len(skipped)}개, 렌더 {outline}",
        f"- {OBJECTIVE}",
        "- 불일치 비율은 불일치 픽셀 수를 실기 crop의 잉크 픽셀 수로 "
        "나눈 값이다.",
        f"- 최적 오프셋이 탐색 반경 경계에 걸린 fit은 {at_boundary}개"
        f"/{len(fits)}개다. 많으면 기준 위치 추정이나 반경을 다시 본다.",
    ]
    for bucket, entries in rankings.items():
        title = "전체" if bucket == "all" else f"s {bucket} 구간"
        sections.extend(["", f"## {title}: 상위 파이프라인", ""])
        sections.append(
            markdown_table(
                [
                    "순위",
                    "파이프라인",
                    "샘플",
                    "평균 불일치 비율",
                    "평균 MAE",
                    "불일치/잉크",
                ],
                [
                    [
                        str(index + 1),
                        f"`{entry.pipeline}`",
                        str(entry.samples),
                        percent(entry.mean_ratio),
                        f"{entry.mean_mae:.2f}",
                        f"{entry.mismatch}/{entry.ink}",
                    ]
                    for index, entry in enumerate(entries)
                ],
            )
        )
        if bucket == "all" or not entries:
            continue
        leader = entries[0].pipeline
        rows = []
        for sample in sorted(samples, key=lambda item: item.id):
            if sample.bucket != bucket or sample.id not in by_sample:
                continue
            own = min(by_sample[sample.id], key=lambda f: (f.mismatch, f.mae))
            lead = next(
                fit for fit in by_sample[sample.id] if fit.pipeline == leader
            )
            rows.append(
                [
                    sample.id,
                    sample.screen,
                    sample.text,
                    f"{sample.scale:.2f}",
                    f"`{own.pipeline}`",
                    f"({own.offset[0]}, {own.offset[1]})",
                    f"{own.mismatch} ({percent(own.ratio)})",
                    f"{own.mae:.2f}",
                    f"({lead.offset[0]}, {lead.offset[1]})",
                    f"{lead.mismatch} ({percent(lead.ratio)})",
                    f"{lead.mae:.2f}",
                ]
            )
        sections.extend(["", f"### s {bucket} 구간: 샘플별 최적", ""])
        sections.append(
            markdown_table(
                [
                    "샘플",
                    "화면",
                    "닉네임",
                    "s",
                    "샘플 최적 파이프라인",
                    "오프셋",
                    "불일치 (비율)",
                    "MAE",
                    "1위 오프셋",
                    "1위 불일치 (비율)",
                    "1위 MAE",
                ],
                rows,
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
