#!/usr/bin/env python3
"""Validate and summarize the real exact4 B1 FP8 drafter-head timing pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import statistics
from pathlib import Path
from typing import Any

try:
    from .fr13_draft_head_fp8_gate import (
        GRAPH_SIGNATURE,
        _validate_engagement,
    )
except ImportError:
    from fr13_draft_head_fp8_gate import (
        GRAPH_SIGNATURE,
        _validate_engagement,
    )


SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db"
    "4a46ebad5db777c5b999bf797ae853f5"
)
T_CRITICAL_ONE_SIDED_95_DF3 = 2.3533634348018264
MIN_RETAINED_WALL_FRACTION = 0.99
MIN_TASK_COUNTER_STEPS = 64
FLOORS = {
    False: {
        "mandatory_weight_bytes": 32_666_638_208,
        "weight_floor_ms": 119.658015414,
        "one_sided_u95_cap_ms": 137.6067177261,
    },
    True: {
        "mandatory_weight_bytes": 30_989_326_208,
        "weight_floor_ms": 113.514015414,
        "one_sided_u95_cap_ms": 130.541117726,
    },
}
RAW = {
    "drafts": "vllm:spec_decode_num_drafts_total",
    "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
    "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    "fwd_gpu_seconds": "vllm:fr13_decode_forward_gpu_seconds_total",
    "fwd_gpu_steps": "vllm:fr13_decode_forward_gpu_steps_total",
    "fwd_gpu_drafts": "vllm:fr13_decode_forward_gpu_drafts_total",
    "wall_seconds": "vllm:fr13_decode_step_wall_seconds_total",
    "wall_steps": "vllm:fr13_decode_step_wall_steps_total",
    "wall_drafts": "vllm:fr13_decode_step_wall_drafts_total",
    "drafter_gpu_seconds": "vllm:fr13_drafter_gpu_seconds_total",
    "drafter_gpu_spans": "vllm:fr13_drafter_gpu_spans_total",
    "committer_gpu_seconds": "vllm:fr13_committer_gpu_seconds_total",
    "committer_gpu_spans": "vllm:fr13_committer_gpu_spans_total",
}


def _regular(path: Path, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    raw = path.read_bytes()
    payload = json.loads(raw, object_pairs_hook=_reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _finite(record: dict[str, Any], key: str, label: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} is missing or nonnumeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label}.{key} must be finite and positive")
    return result


def _nonnegative(record: dict[str, Any], key: str, label: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} is missing or nonnumeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label}.{key} must be finite and nonnegative")
    return result


def _integral(record: dict[str, Any], key: str, label: str) -> int:
    value = _nonnegative(record, key, label)
    if value != round(value):
        raise ValueError(f"{label}.{key} is not an integer counter")
    return int(value)


def _close(
    actual: float,
    expected: float,
    label: str,
    *,
    absolute: float = 1e-9,
    relative: float = 1e-9,
) -> None:
    if not math.isclose(
        actual, expected, rel_tol=relative, abs_tol=absolute
    ):
        raise ValueError(f"{label} mismatch: {actual} != {expected}")


def _equal_task_u95(values: list[float]) -> dict[str, float]:
    if len(values) != 4:
        raise ValueError("descriptive U95 requires the canonical four tasks")
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    return {
        "mean_ms": mean,
        "sample_standard_deviation_ms": standard_deviation,
        "standard_error_ms": standard_error,
        "critical_value": T_CRITICAL_ONE_SIDED_95_DF3,
        "u95_ms": (
            mean + T_CRITICAL_ONE_SIDED_95_DF3 * standard_error
        ),
    }


def _validate_arm(
    payload: dict[str, Any],
    raw_bytes: bytes,
    *,
    label: str,
    expected_arm: str,
    task_ids: list[str],
    fp8: bool,
) -> dict[str, Any]:
    floor = FLOORS[fp8]
    engagement = payload.get("engagement")
    aggregate = payload.get("raw_counter_delta_aggregate")
    per_task = payload.get("per_task")
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("regime") != "deployment"
        or payload.get("instrument") != "OFF"
        or payload.get("arm") != expected_arm
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != 4
        or sorted(payload.get("task_instance_ids", [])) != task_ids
        or payload.get("draft_vocab_k") != 65_536
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_head_fp8") is not fp8
        or payload.get("floor_is_full_step_hardware_floor") is not False
        or payload.get("floor_reference_scope")
        != "fixed32_mandatory_weight_read_or_row_compute_lower_bound"
        or not isinstance(engagement, dict)
        or engagement.get("engaged") is not True
        or engagement.get("tok_per_draft") != 31.0
        or engagement.get("expected_tok_per_draft") != 31.0
        or not isinstance(aggregate, dict)
        or not isinstance(per_task, list)
        or len(per_task) != 4
    ):
        raise ValueError(f"{label} is not exact4 real SWE-Verified B1")
    if sorted(row.get("instance_id") for row in per_task) != task_ids:
        raise ValueError(f"{label} per-task identity drifted")

    task_fields = {
        "drafts": RAW["drafts"],
        "accepted_tokens": RAW["accepted_tokens"],
        "fwd_gpu_seconds": RAW["fwd_gpu_seconds"],
        "fwd_gpu_steps": RAW["fwd_gpu_steps"],
        "fwd_gpu_drafts": RAW["fwd_gpu_drafts"],
        "wall_seconds": RAW["wall_seconds"],
        "wall_steps": RAW["wall_steps"],
        "wall_drafts": RAW["wall_drafts"],
        "drafter_gpu_seconds": RAW["drafter_gpu_seconds"],
        "drafter_gpu_spans": RAW["drafter_gpu_spans"],
        "committer_gpu_seconds": RAW["committer_gpu_seconds"],
        "committer_gpu_spans": RAW["committer_gpu_spans"],
    }
    task_sums = {metric: 0.0 for metric in task_fields.values()}
    task_wall_ms: dict[str, float] = {}
    task_acceptance: dict[str, float] = {}
    retained_fractions: dict[str, float] = {}
    for row in per_task:
        task_id = row["instance_id"]
        task_label = f"{label}:{task_id}"
        fwd_steps = _integral(row, "fwd_gpu_steps", task_label)
        fwd_drafts = _integral(row, "fwd_gpu_drafts", task_label)
        wall_steps = _integral(row, "wall_steps", task_label)
        wall_drafts = _integral(row, "wall_drafts", task_label)
        drafts = _integral(row, "drafts", task_label)
        accepted = _nonnegative(row, "accepted_tokens", task_label)
        if (
            fwd_steps < MIN_TASK_COUNTER_STEPS
            or fwd_drafts != fwd_steps
            or wall_steps != wall_drafts
            or wall_steps > fwd_steps
            or wall_steps / fwd_steps < MIN_RETAINED_WALL_FRACTION
            or drafts < fwd_drafts
            or row.get("tok_per_draft") != 31.0
        ):
            raise ValueError(f"{task_label} counter window is invalid")
        wall_seconds = _finite(row, "wall_seconds", task_label)
        task_wall_ms[task_id] = wall_seconds / wall_drafts * 1000.0
        task_acceptance[task_id] = accepted / drafts
        retained_fractions[task_id] = wall_steps / fwd_steps
        for field, metric in task_fields.items():
            task_sums[metric] += _nonnegative(row, field, task_label)

    for metric, task_sum in task_sums.items():
        _close(
            _nonnegative(aggregate, metric, f"{label}:aggregate"),
            task_sum,
            f"{label} per-task sum {metric}",
            absolute=1e-7,
        )
    aggregate_drafts = _integral(aggregate, RAW["drafts"], label)
    aggregate_draft_tokens = _integral(
        aggregate, RAW["draft_tokens"], label
    )
    aggregate_accepted = _nonnegative(
        aggregate, RAW["accepted_tokens"], label
    )
    aggregate_fwd_steps = _integral(
        aggregate, RAW["fwd_gpu_steps"], label
    )
    aggregate_fwd_drafts = _integral(
        aggregate, RAW["fwd_gpu_drafts"], label
    )
    aggregate_wall_steps = _integral(
        aggregate, RAW["wall_steps"], label
    )
    aggregate_wall_drafts = _integral(
        aggregate, RAW["wall_drafts"], label
    )
    if (
        aggregate_drafts <= 0
        or aggregate_draft_tokens != aggregate_drafts * 31
        or aggregate_fwd_steps < 4 * MIN_TASK_COUNTER_STEPS
        or aggregate_fwd_drafts != aggregate_fwd_steps
        or aggregate_wall_steps != aggregate_wall_drafts
        or aggregate_wall_steps > aggregate_fwd_steps
        or aggregate_wall_steps / aggregate_fwd_steps
        < MIN_RETAINED_WALL_FRACTION
    ):
        raise ValueError(f"{label} aggregate counter window is invalid")

    values = {
        key: _finite(payload, key, label)
        for key in (
            "accept_per_event",
            "committed_per_event",
            "committer_gpu_ms_per_step",
            "compute_floor_ms",
            "derived_tps_fullstep_gpu",
            "drafter_gpu_ms_per_step",
            "events_per_step",
            "floor_ms",
            "floor_ratio",
            "fullstep_alignment_ratio",
            "measured_tps_fullstep_wall",
            "rows_per_step",
            "s_per_fwd_gpu",
            "step_wall_ms",
            "wall_s_per_event",
            "wall_steps_measured",
            "weight_floor_ms",
        )
    }
    if (
        payload.get("mandatory_weight_bytes")
        != floor["mandatory_weight_bytes"]
        or payload.get("weight_floor_bandwidth_bytes_per_s")
        != 273_000_000_000
    ):
        raise ValueError(f"{label} mandatory-byte identity drifted")
    _close(values["events_per_step"], 1.0, f"{label} events per step")
    _close(values["rows_per_step"], 32.0, f"{label} rows per step")
    _close(values["compute_floor_ms"], 17.28, f"{label} compute floor")
    _close(
        values["weight_floor_ms"],
        floor["weight_floor_ms"],
        f"{label} weight floor",
    )
    _close(
        values["floor_ms"],
        floor["weight_floor_ms"],
        f"{label} active floor",
    )
    _close(
        values["committed_per_event"],
        values["accept_per_event"] + 1.0,
        f"{label} committed tokens",
    )
    _close(
        values["accept_per_event"],
        aggregate_accepted / aggregate_drafts,
        f"{label} acceptance",
    )
    _close(
        values["step_wall_ms"],
        values["wall_s_per_event"] * 1000.0,
        f"{label} full step wall",
    )
    _close(
        values["measured_tps_fullstep_wall"],
        values["committed_per_event"] / values["wall_s_per_event"],
        f"{label} full wall TPS",
        absolute=1e-8,
    )
    _close(
        values["floor_ratio"],
        values["step_wall_ms"] / values["floor_ms"],
        f"{label} floor ratio",
    )
    _close(
        values["s_per_fwd_gpu"],
        _nonnegative(aggregate, RAW["fwd_gpu_seconds"], label)
        / aggregate_fwd_drafts,
        f"{label} SFWD",
    )
    drafter_spans = _integral(
        aggregate, RAW["drafter_gpu_spans"], label
    )
    committer_spans = _integral(
        aggregate, RAW["committer_gpu_spans"], label
    )
    if drafter_spans == 0 or committer_spans == 0:
        raise ValueError(f"{label} component timer spans are empty")
    _close(
        values["drafter_gpu_ms_per_step"],
        _nonnegative(aggregate, RAW["drafter_gpu_seconds"], label)
        / drafter_spans
        * 1000.0,
        f"{label} DFWD",
    )
    _close(
        values["committer_gpu_ms_per_step"],
        _nonnegative(aggregate, RAW["committer_gpu_seconds"], label)
        / committer_spans
        * 1000.0,
        f"{label} CFWD",
    )
    _close(
        values["wall_steps_measured"],
        float(aggregate_wall_steps),
        f"{label} wall sample count",
    )
    sfwd_ms = values["s_per_fwd_gpu"] * 1000.0
    full_gpu_ms = (
        sfwd_ms
        + values["drafter_gpu_ms_per_step"]
        + values["committer_gpu_ms_per_step"]
    )
    _close(
        values["derived_tps_fullstep_gpu"],
        values["committed_per_event"] / (full_gpu_ms / 1000.0),
        f"{label} full GPU TPS",
        absolute=1e-8,
    )
    task_u95 = _equal_task_u95(list(task_wall_ms.values()))
    task_u95["floor_ratio"] = task_u95["u95_ms"] / values["floor_ms"]
    task_u95["cap_ms"] = floor["one_sided_u95_cap_ms"]
    task_u95["descriptive_screen_pass"] = (
        task_u95["u95_ms"] <= floor["one_sided_u95_cap_ms"]
    )
    return {
        "raw_sha256": _sha256(raw_bytes),
        "mandatory_weight_bytes": floor["mandatory_weight_bytes"],
        "floor_ms": values["floor_ms"],
        "floor_ratio": values["floor_ratio"],
        "one_sided_u95_cap_ms": floor["one_sided_u95_cap_ms"],
        "step_wall_ms": values["step_wall_ms"],
        "measured_tps_fullstep_wall": values[
            "measured_tps_fullstep_wall"
        ],
        "accept_per_event": values["accept_per_event"],
        "committed_per_event": values["committed_per_event"],
        "wall_steps_measured": aggregate_wall_steps,
        "pure_decode_steps_measured": aggregate_fwd_steps,
        "retained_wall_fraction": (
            aggregate_wall_steps / aggregate_fwd_steps
        ),
        "per_task_retained_wall_fraction": retained_fractions,
        "per_task_wall_ms": task_wall_ms,
        "per_task_accept_per_event": task_acceptance,
        "descriptive_equal_task_one_sided_u95": task_u95,
        "gpu_components_ms_per_step": {
            "sfwd": sfwd_ms,
            "dfwd": values["drafter_gpu_ms_per_step"],
            "cfwd": values["committer_gpu_ms_per_step"],
            "total": full_gpu_ms,
        },
        "derived_tps_fullstep_gpu": values["derived_tps_fullstep_gpu"],
        "wall_residual_ms": values["step_wall_ms"] - full_gpu_ms,
        "fullstep_alignment_ratio": values["fullstep_alignment_ratio"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-engagement", type=Path, required=True)
    parser.add_argument("--promotion-credential", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--stock-arm", required=True)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--stock-fa2-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    _regular(args.subset, "canonical exact4 subset")
    subset_raw = args.subset.read_bytes()
    if _sha256(subset_raw) != SUBSET_SHA256:
        raise ValueError("canonical exact4 subset SHA-256 drifted")
    subset = json.loads(subset_raw, object_pairs_hook=_reject_duplicates)
    task_ids = sorted(subset.get("instance_ids", []))
    if len(task_ids) != 4 or len(set(task_ids)) != 4:
        raise ValueError("canonical exact4 task identity drifted")

    stock, stock_raw = _load(args.stock, "stock deploy-speed")
    candidate, candidate_raw = _load(
        args.candidate, "candidate deploy-speed"
    )
    engagement, engagement_raw = _load(
        args.candidate_engagement, "candidate FP8 engagement"
    )
    promotion, promotion_raw = _load(
        args.promotion_credential, "FP8 promotion credential"
    )
    if (
        promotion.get("schema")
        != "fr13.fixed32.draft_head_fp8_promotion_credential.v1"
        or promotion.get("status") != "PASS"
        or promotion.get("performance_tuning_eligible") is not True
        or promotion.get("formal_floor_acceptance_eligible") is not False
        or promotion.get("source_commit") != args.expected_source_commit
        or promotion.get("candidate_source_sha256")
        != args.expected_source_sha256
        or promotion.get("engagement")
        != {
            "selected_root_calls": 1,
            "captured_loop_calls": 4,
            "fallback_calls": 0,
            "proposal_logits_source": "fp8_output_direct",
            "bf16_shadow_calls": 0,
            "steady_state_synchronizations": 0,
        }
    ):
        raise ValueError("FP8 promotion credential drifted")
    _validate_engagement(
        engagement,
        source_sha=args.expected_source_sha256,
        source_commit=args.expected_source_commit,
        expected_arm=args.candidate_arm,
    )
    if engagement.get("drafter_graph_signature") != GRAPH_SIGNATURE:
        raise ValueError("candidate graph signature drifted")

    stock_summary = _validate_arm(
        stock,
        stock_raw,
        label="stock",
        expected_arm=args.stock_arm,
        task_ids=task_ids,
        fp8=False,
    )
    candidate_summary = _validate_arm(
        candidate,
        candidate_raw,
        label="candidate",
        expected_arm=args.candidate_arm,
        task_ids=task_ids,
        fp8=True,
    )
    stock_components = stock_summary["gpu_components_ms_per_step"]
    candidate_components = candidate_summary["gpu_components_ms_per_step"]
    result = {
        "schema": "fr13.fixed32.draft_head_fp8_exact4_b1_timing.v1",
        "status": "COMPLETE",
        "classification": "real_swe_verified_exact4_b1_timing_pair",
        "timing_eligible": True,
        "formal_floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "only_arm_delta": "FR13_DRAFT_HEAD_FP8_0_to_1",
        "batch_size": 1,
        "concurrency": 1,
        "task_instance_ids": task_ids,
        "source_commit": args.expected_source_commit,
        "candidate_source_sha256": args.expected_source_sha256,
        "evidence_sha256": {
            "subset": _sha256(subset_raw),
            "stock_deploy_speed": _sha256(stock_raw),
            "candidate_deploy_speed": _sha256(candidate_raw),
            "candidate_engagement": _sha256(engagement_raw),
            "promotion_credential": _sha256(promotion_raw),
            "stock_fa2": args.stock_fa2_sha256,
        },
        "stock": stock_summary,
        "candidate": candidate_summary,
        "delta_candidate_minus_stock": {
            "step_wall_ms": (
                candidate_summary["step_wall_ms"]
                - stock_summary["step_wall_ms"]
            ),
            "step_wall_percent": (
                candidate_summary["step_wall_ms"]
                / stock_summary["step_wall_ms"]
                - 1.0
            )
            * 100.0,
            "measured_tps_fullstep_wall": (
                candidate_summary["measured_tps_fullstep_wall"]
                - stock_summary["measured_tps_fullstep_wall"]
            ),
            "accept_per_event": (
                candidate_summary["accept_per_event"]
                - stock_summary["accept_per_event"]
            ),
            "sfwd_gpu_ms_per_step": (
                candidate_components["sfwd"] - stock_components["sfwd"]
            ),
            "dfwd_gpu_ms_per_step": (
                candidate_components["dfwd"] - stock_components["dfwd"]
            ),
            "cfwd_gpu_ms_per_step": (
                candidate_components["cfwd"] - stock_components["cfwd"]
            ),
            "total_gpu_ms_per_step": (
                candidate_components["total"] - stock_components["total"]
            ),
            "wall_residual_ms": (
                candidate_summary["wall_residual_ms"]
                - stock_summary["wall_residual_ms"]
            ),
        },
        "statistical_scope_note": (
            "The equal-task one-sided U95 is descriptive for this exact4 "
            "promotion screen. Formal Tail23/Hydra27 acceptance remains separate."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(args.out)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
