from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fr10_equivalence_gate import (
    GateReport,
    GateThresholds,
    SamplingRecord,
    StateParityRow,
    TokenRecord,
    compare_records,
    evaluate_state_parity,
    sampling_distribution_distance,
    summarize_sampling_distance,
)


NAIVE_MTP = "naive_mtp"
NON_MTP = "non_mtp"
TREE_MTP = "tree_mtp"
VALID_DECODE_MODES = frozenset({NAIVE_MTP, NON_MTP, TREE_MTP})
MODE_ARG_KEYS = ("fr10_decode_mode", "decode_mode", "mode")


def normalize_decode_mode(value: Any, *, default: str = TREE_MTP) -> str:
    if value is None or value == "":
        mode = default
    elif isinstance(value, str):
        mode = value.strip().lower().replace("-", "_")
    else:
        raise ValueError(f"FR10 decode mode must be a string, got {type(value).__name__}")
    if mode not in VALID_DECODE_MODES:
        allowed = ", ".join(sorted(VALID_DECODE_MODES))
        raise ValueError(f"invalid FR10 decode mode {mode!r}; allowed: {allowed}")
    return mode


def decode_mode_from_extra_args(
    extra_args: Mapping[str, Any] | None,
    *,
    default: str = TREE_MTP,
) -> str:
    if not extra_args:
        return normalize_decode_mode(None, default=default)
    for key in MODE_ARG_KEYS:
        if key in extra_args:
            return normalize_decode_mode(extra_args[key], default=default)
    return normalize_decode_mode(None, default=default)


def decode_mode_from_sampling_params(
    sampling_params: Any,
    *,
    default: str = TREE_MTP,
) -> str:
    extra_args = getattr(sampling_params, "extra_args", None)
    return decode_mode_from_extra_args(extra_args, default=default)


def decode_mode_from_request(request: Any, *, default: str = TREE_MTP) -> str:
    return decode_mode_from_sampling_params(
        getattr(request, "sampling_params", None), default=default
    )


def mode_allows_spec_decode(mode: str) -> bool:
    return normalize_decode_mode(mode) != NON_MTP


def mode_uses_tree_gdn(mode: str) -> bool:
    return normalize_decode_mode(mode) == TREE_MTP


def tree_path0_choice_indices(tree_choices: Sequence[Sequence[int]]) -> tuple[int, ...]:
    path0 = [
        (idx, tuple(choice))
        for idx, choice in enumerate(tree_choices)
        if len(choice) > 0 and all(int(part) == 0 for part in choice)
    ]
    path0.sort(key=lambda row: len(row[1]))
    return tuple(idx for idx, _ in path0)


def select_path0_spec_tokens(
    spec_token_ids: Sequence[int],
    tree_choices: Sequence[Sequence[int]],
) -> list[int]:
    indices = tree_path0_choice_indices(tree_choices)
    if not indices:
        return list(spec_token_ids)
    return [int(spec_token_ids[idx]) for idx in indices if idx < len(spec_token_ids)]


@dataclass(frozen=True)
class ModeBatchDecision:
    mode: str | None
    accepted_indices: tuple[int, ...]
    deferred_indices: tuple[int, ...]
    mixed: bool


def homogeneous_mode_decision(modes: Sequence[str]) -> ModeBatchDecision:
    if not modes:
        return ModeBatchDecision(None, (), (), False)
    first = normalize_decode_mode(modes[0])
    accepted: list[int] = []
    deferred: list[int] = []
    for idx, mode_value in enumerate(modes):
        mode = normalize_decode_mode(mode_value)
        if mode == first:
            accepted.append(idx)
        else:
            deferred.append(idx)
    return ModeBatchDecision(
        mode=first,
        accepted_indices=tuple(accepted),
        deferred_indices=tuple(deferred),
        mixed=bool(deferred),
    )


def require_homogeneous_modes(modes: Iterable[str]) -> str | None:
    modes_tuple = tuple(modes)
    decision = homogeneous_mode_decision(modes_tuple)
    if decision.mixed:
        observed = ", ".join(modes_tuple)
        raise RuntimeError(f"FR10 mixed decode-mode batch rejected: {observed}")
    return decision.mode


def evaluate_relaunch_equivalence_tokens(
    shared: Mapping[tuple[str, int], TokenRecord],
    dedicated: Mapping[tuple[str, int], TokenRecord],
    *,
    mode: str,
    batches: set[str] | None = None,
) -> GateReport:
    report = GateReport(passed=True)
    mode = normalize_decode_mode(mode)
    flips = compare_records(dedicated, shared, batches=batches)
    report.metrics.update(
        {
            "mode": mode,
            "flips": [flip.__dict__ for flip in flips],
            "flip_count": len(flips),
        }
    )
    if flips:
        report.add_violation(
            f"shared-server {mode} differs from dedicated-relaunch {mode}: "
            f"{len(flips)} token flips"
        )
    return report


def evaluate_relaunch_equivalence_sampling(
    shared: Sequence[SamplingRecord],
    dedicated: Sequence[SamplingRecord],
    same_regime_floor: Sequence[SamplingRecord],
    same_regime_reference: Sequence[SamplingRecord],
    *,
    mode: str,
    positions: int = 12,
) -> GateReport:
    report = GateReport(passed=True)
    mode = normalize_decode_mode(mode)
    cross_summary = summarize_sampling_distance(
        sampling_distribution_distance(dedicated, shared, positions=positions)
    )
    same_summary = summarize_sampling_distance(
        sampling_distribution_distance(
            same_regime_reference, same_regime_floor, positions=positions
        )
    )
    cross_tv = float(cross_summary["max_aggregate_tv"])
    same_tv = float(same_summary["max_aggregate_tv"])
    report.metrics.update(
        {
            "mode": mode,
            "cross": cross_summary,
            "same_regime_floor": same_summary,
        }
    )
    if cross_tv > same_tv:
        report.add_violation(
            f"shared-server {mode} sampling TV {cross_tv:.6g} exceeds "
            f"same-regime floor {same_tv:.6g}"
        )
    return report


def evaluate_mixed_mode_contamination_control(
    rows: Sequence[StateParityRow],
    *,
    thresholds: GateThresholds = GateThresholds(),
) -> GateReport:
    parity = evaluate_state_parity(rows, thresholds=thresholds)
    report = GateReport(passed=not parity.passed)
    report.metrics.update(parity.metrics)
    report.metrics["violations_observed"] = list(parity.violations)
    if parity.passed:
        report.add_violation(
            "mixed-mode negative control did not contaminate state parity; "
            "isolation test has no power"
        )
    return report

