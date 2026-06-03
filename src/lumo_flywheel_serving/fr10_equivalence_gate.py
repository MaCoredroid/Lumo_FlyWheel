from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RecordKey = tuple[str, int]


@dataclass(frozen=True)
class TokenRecord:
    batch: str
    choice_index: int
    prompt: str
    token_ids: tuple[int, ...]
    text: str = ""

    @property
    def key(self) -> RecordKey:
        return (self.batch, self.choice_index)


@dataclass(frozen=True)
class TokenFlip:
    batch: str
    choice_index: int
    prompt: str
    position: int
    reference_token: int | None
    candidate_token: int | None
    reference_len: int
    candidate_len: int
    margin: float | None = None

    @property
    def key(self) -> RecordKey:
        return (self.batch, self.choice_index)


@dataclass(frozen=True)
class FlipDistribution:
    flips: int
    total_records: int
    rate: float
    position_counts: dict[int, int]
    max_margin: float | None
    mean_margin: float | None


@dataclass(frozen=True)
class StateParityRow:
    layer: int
    node: int
    max_state_abs: float
    max_output_abs: float = 0.0
    position: int | None = None
    tag: str = "clean"


@dataclass(frozen=True)
class NegativeControl:
    name: str
    observed_delta: float
    must_exceed: float


@dataclass(frozen=True)
class GateThresholds:
    state_abs: float = 2e-5
    output_abs: float = 2e-5
    margin_indifference: float = 6e-5
    flip_rate_slack: float = 1 / 16
    position_tv_slack: float = 0.25
    chi_square_slack: float = 8.0
    accumulation_slope_abs: float = 1e-6


@dataclass
class GateReport:
    passed: bool
    violations: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def add_violation(self, message: str) -> None:
        self.passed = False
        self.violations.append(message)


def load_token_artifact(path: str | Path) -> dict[RecordKey, TokenRecord]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = data.get("records", data if isinstance(data, list) else [])
    out: dict[RecordKey, TokenRecord] = {}
    for row in records:
        record = TokenRecord(
            batch=str(row["batch"]),
            choice_index=int(row["choice_index"]),
            prompt=str(row.get("prompt", "")),
            token_ids=tuple(int(x) for x in row.get("token_ids", [])),
            text=str(row.get("text", "")),
        )
        out[record.key] = record
    return out


def first_token_diff(left: Sequence[int], right: Sequence[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if int(left_id) != int(right_id):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_records(
    reference: Mapping[RecordKey, TokenRecord],
    candidate: Mapping[RecordKey, TokenRecord],
    *,
    batches: set[str] | None = None,
) -> list[TokenFlip]:
    flips: list[TokenFlip] = []
    for key in sorted(set(reference) & set(candidate)):
        if batches is not None and key[0] not in batches:
            continue
        left = reference[key]
        right = candidate[key]
        pos = first_token_diff(left.token_ids, right.token_ids)
        if pos is None:
            continue
        flips.append(
            TokenFlip(
                batch=key[0],
                choice_index=key[1],
                prompt=left.prompt,
                position=pos,
                reference_token=left.token_ids[pos] if pos < len(left.token_ids) else None,
                candidate_token=right.token_ids[pos] if pos < len(right.token_ids) else None,
                reference_len=len(left.token_ids),
                candidate_len=len(right.token_ids),
            )
        )
    return flips


def matched_record_count(
    reference: Mapping[RecordKey, TokenRecord],
    candidate: Mapping[RecordKey, TokenRecord],
    *,
    batches: set[str] | None = None,
) -> int:
    keys = set(reference) & set(candidate)
    if batches is not None:
        keys = {key for key in keys if key[0] in batches}
    return len(keys)


def flip_distribution(flips: Sequence[TokenFlip], total_records: int) -> FlipDistribution:
    margins = [flip.margin for flip in flips if flip.margin is not None]
    return FlipDistribution(
        flips=len(flips),
        total_records=total_records,
        rate=(len(flips) / total_records) if total_records else 0.0,
        position_counts=dict(Counter(flip.position for flip in flips)),
        max_margin=max(margins) if margins else None,
        mean_margin=(sum(margins) / len(margins)) if margins else None,
    )


def total_variation(left_counts: Mapping[int, int], right_counts: Mapping[int, int]) -> float:
    keys = set(left_counts) | set(right_counts)
    left_total = sum(left_counts.values())
    right_total = sum(right_counts.values())
    if left_total == 0 and right_total == 0:
        return 0.0
    tv = 0.0
    for key in keys:
        left_p = left_counts.get(key, 0) / left_total if left_total else 0.0
        right_p = right_counts.get(key, 0) / right_total if right_total else 0.0
        tv += abs(left_p - right_p)
    return 0.5 * tv


def chi_square_distance(
    observed_counts: Mapping[int, int],
    expected_counts: Mapping[int, int],
    *,
    floor: float = 1e-9,
) -> float:
    keys = set(observed_counts) | set(expected_counts)
    observed_total = sum(observed_counts.values())
    expected_total = sum(expected_counts.values())
    if observed_total == 0 and expected_total == 0:
        return 0.0
    stat = 0.0
    for key in keys:
        expected_p = (
            expected_counts.get(key, 0) / expected_total if expected_total else floor
        )
        expected = max(floor, expected_p * max(1, observed_total))
        observed = observed_counts.get(key, 0)
        stat += (observed - expected) ** 2 / expected
    return stat


def evaluate_flip_distribution_equivalence(
    *,
    tree_flips: Sequence[TokenFlip],
    baseline_flips: Sequence[TokenFlip],
    total_records: int,
    thresholds: GateThresholds = GateThresholds(),
) -> GateReport:
    report = GateReport(passed=True)
    tree_dist = flip_distribution(tree_flips, total_records)
    baseline_dist = flip_distribution(baseline_flips, total_records)
    tv = total_variation(tree_dist.position_counts, baseline_dist.position_counts)
    chi2 = chi_square_distance(tree_dist.position_counts, baseline_dist.position_counts)
    report.metrics.update(
        {
            "tree": tree_dist.__dict__,
            "baseline": baseline_dist.__dict__,
            "position_tv": tv,
            "position_chi_square": chi2,
        }
    )
    if tree_dist.rate > baseline_dist.rate + thresholds.flip_rate_slack:
        report.add_violation(
            f"tree flip rate {tree_dist.rate:.6g} exceeds baseline {baseline_dist.rate:.6g}"
        )
    if tv > thresholds.position_tv_slack:
        report.add_violation(f"tree flip position TV {tv:.6g} exceeds threshold")
    if chi2 > thresholds.chi_square_slack:
        report.add_violation(f"tree flip position chi-square {chi2:.6g} exceeds threshold")
    return report


def attach_flip_margins(
    flips: Iterable[TokenFlip],
    margins: Mapping[RecordKey, Mapping[int, float]],
) -> list[TokenFlip]:
    out: list[TokenFlip] = []
    for flip in flips:
        margin = margins.get(flip.key, {}).get(flip.position)
        out.append(
            TokenFlip(
                batch=flip.batch,
                choice_index=flip.choice_index,
                prompt=flip.prompt,
                position=flip.position,
                reference_token=flip.reference_token,
                candidate_token=flip.candidate_token,
                reference_len=flip.reference_len,
                candidate_len=flip.candidate_len,
                margin=margin,
            )
        )
    return out


def evaluate_flip_margins(
    flips: Sequence[TokenFlip],
    *,
    thresholds: GateThresholds = GateThresholds(),
    require_margins: bool = True,
) -> GateReport:
    report = GateReport(passed=True)
    margins = [flip.margin for flip in flips if flip.margin is not None]
    missing = len(flips) - len(margins)
    report.metrics.update(
        {
            "flips": len(flips),
            "missing_margins": missing,
            "max_margin": max(margins) if margins else None,
            "mean_margin": (sum(margins) / len(margins)) if margins else None,
        }
    )
    if require_margins and missing:
        report.add_violation(f"{missing} flips missing logit-margin evidence")
    for flip in flips:
        if flip.margin is not None and flip.margin > thresholds.margin_indifference:
            report.add_violation(
                "high-margin flip "
                f"batch={flip.batch} choice={flip.choice_index} pos={flip.position} "
                f"margin={flip.margin:.6g}"
            )
    return report


def evaluate_state_parity(
    rows: Sequence[StateParityRow],
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> GateReport:
    report = GateReport(passed=True)
    max_state = max((row.max_state_abs for row in rows), default=0.0)
    max_output = max((row.max_output_abs for row in rows), default=0.0)
    report.metrics.update(
        {
            "rows": len(rows),
            "max_state_abs": max_state,
            "max_output_abs": max_output,
        }
    )
    for row in rows:
        if row.max_state_abs > thresholds.state_abs:
            report.add_violation(
                f"state parity fail layer={row.layer} node={row.node} "
                f"delta={row.max_state_abs:.6g}"
            )
        if row.max_output_abs > thresholds.output_abs:
            report.add_violation(
                f"output parity fail layer={row.layer} node={row.node} "
                f"delta={row.max_output_abs:.6g}"
            )
    return report


def evaluate_accumulation(
    rows: Sequence[StateParityRow],
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> GateReport:
    report = GateReport(passed=True)
    positioned = [row for row in rows if row.position is not None]
    if len(positioned) < 2:
        report.metrics["slope"] = 0.0
        return report
    xs = [float(row.position) for row in positioned]
    ys = [float(row.max_state_abs) for row in positioned]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    slope = 0.0 if denom == 0.0 else sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    report.metrics["slope"] = slope
    if abs(slope) > thresholds.accumulation_slope_abs:
        report.add_violation(f"state divergence accumulates with slope {slope:.6g}")
    return report


def evaluate_negative_controls(controls: Sequence[NegativeControl]) -> GateReport:
    report = GateReport(passed=True)
    report.metrics["controls"] = [control.__dict__ for control in controls]
    for control in controls:
        if control.observed_delta <= control.must_exceed:
            report.add_violation(
                f"negative control {control.name} did not fail loudly: "
                f"{control.observed_delta:.6g} <= {control.must_exceed:.6g}"
            )
    return report


def merge_reports(*reports: GateReport) -> GateReport:
    merged = GateReport(passed=all(report.passed for report in reports))
    for idx, report in enumerate(reports):
        merged.metrics[f"report_{idx}"] = report.metrics
        merged.violations.extend(report.violations)
    return merged


def evaluate_three_way_gate(
    *,
    non_mtp: Mapping[RecordKey, TokenRecord],
    naive_mtp: Mapping[RecordKey, TokenRecord],
    tree_mtp: Mapping[RecordKey, TokenRecord],
    state_rows: Sequence[StateParityRow],
    negative_controls: Sequence[NegativeControl],
    batches: set[str] | None = None,
    tree_flip_margins: Mapping[RecordKey, Mapping[int, float]] | None = None,
    baseline_flip_margins: Mapping[RecordKey, Mapping[int, float]] | None = None,
    thresholds: GateThresholds = GateThresholds(),
    require_margins: bool = True,
) -> GateReport:
    total = min(
        matched_record_count(non_mtp, naive_mtp, batches=batches),
        matched_record_count(non_mtp, tree_mtp, batches=batches),
    )
    baseline_flips = compare_records(non_mtp, naive_mtp, batches=batches)
    tree_flips = compare_records(non_mtp, tree_mtp, batches=batches)
    if baseline_flip_margins is not None:
        baseline_flips = attach_flip_margins(baseline_flips, baseline_flip_margins)
    if tree_flip_margins is not None:
        tree_flips = attach_flip_margins(tree_flips, tree_flip_margins)
    reports = [
        evaluate_flip_distribution_equivalence(
            tree_flips=tree_flips,
            baseline_flips=baseline_flips,
            total_records=total,
            thresholds=thresholds,
        ),
        evaluate_flip_margins(tree_flips, thresholds=thresholds, require_margins=require_margins),
        evaluate_flip_margins(
            baseline_flips,
            thresholds=thresholds,
            require_margins=require_margins,
        ),
        evaluate_state_parity(state_rows, thresholds=thresholds),
        evaluate_accumulation(state_rows, thresholds=thresholds),
        evaluate_negative_controls(negative_controls),
    ]
    report = merge_reports(*reports)
    report.metrics["summary"] = {
        "matched_records": total,
        "baseline_flips_vs_non_mtp": len(baseline_flips),
        "tree_flips_vs_non_mtp": len(tree_flips),
        "batches": sorted(batches) if batches is not None else None,
    }
    return report


def default_fr10_negative_controls() -> tuple[NegativeControl, ...]:
    return (
        NegativeControl("linear-mask-sibling-leak", observed_delta=0.0173, must_exceed=2e-5),
        NegativeControl("shared-parent-contamination", observed_delta=5.7e-4, must_exceed=2e-5),
        NegativeControl("native-linear-on-tree-contamination", observed_delta=0.56, must_exceed=2e-5),
    )
