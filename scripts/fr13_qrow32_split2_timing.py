#!/usr/bin/env python3
"""Reduce the PASS-gated qrow32 no-split exact4 timing screen.

The historical filename is retained because the composed timing reducer imports
this module, but every accepted and emitted contract is explicitly no-split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from fr13_qrow32_b1_pass_sidecar import (
    ARM,
    CANDIDATE_SHA256,
    CANDIDATE_SIZE,
    EXACT4_SUBSET_SHA256,
    EXACT4_TASK_IDS,
    FA2_HEAD,
    SOURCE_CLOSURE_SHA256,
)


SCHEMA = "fr13.fixed32.fa2_qrow32_nosplit.exact4_timing.v1"
MEASURE_SCHEMA = "fr13.measure.deploy_speed.v1"
ENGAGEMENT_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_production_engagement.v2"
T_CRITICAL_ONE_SIDED_95_DF3 = 2.3533634348018264
MIN_WALL_STEPS_PER_TASK = 64
MIN_RETAINED_WALL_FRACTION = 0.99


def _load(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("ascii"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload, raw


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} is not finite and valid")
    return result


def _integer(value: Any, label: str) -> int:
    number = _number(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} is not integral")
    return int(number)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def equal_task_u95(values: list[float]) -> dict[str, float]:
    if len(values) != 4:
        raise ValueError("exact4 U95 requires four equal-weight task observations")
    mean = statistics.fmean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(4.0)
    return {
        "mean_ms": mean,
        "sample_sd_ms": sample_sd,
        "standard_error_ms": standard_error,
        "critical_value": T_CRITICAL_ONE_SIDED_95_DF3,
        "degrees_of_freedom": 3,
        "u95_ms": mean + T_CRITICAL_ONE_SIDED_95_DF3 * standard_error,
    }


def reduce_timing(
    *,
    subset_path: Path,
    measure_path: Path,
    baseline_path: Path,
    engagement_path: Path,
    health_path: Path,
    traffic_audit_path: Path,
    source_commit: str,
    patch_source_sha256: str,
    pass_sha256: str,
    pass_sidecar_sha256: str,
    runner_sha256: str,
    block_map_sha256: str,
    floor_ms: float,
    cap_ms: float,
    arm: str,
) -> dict[str, Any]:
    subset, subset_raw = _load(subset_path)
    measure, measure_raw = _load(measure_path)
    baseline, baseline_raw = _load(baseline_path)
    engagement, engagement_raw = _load(engagement_path)
    health, health_raw = _load(health_path)
    audit, audit_raw = _load(traffic_audit_path)
    task_ids = sorted(EXACT4_TASK_IDS)
    if (
        _sha(subset_raw) != EXACT4_SUBSET_SHA256
        or sorted(subset.get("instance_ids", [])) != task_ids
    ):
        raise ValueError("timing subset is not canonical exact4")
    if (
        measure.get("schema") != MEASURE_SCHEMA
        or measure.get("instrument") != "OFF"
        or measure.get("regime") != "deployment"
        or measure.get("arm") != arm
        or measure.get("batch_size") != 1
        or measure.get("n_tasks") != 4
        or sorted(measure.get("task_instance_ids", [])) != task_ids
        or measure.get("draft_vocab_root") != 1
        or measure.get("draft_vocab_k") != 65536
        or measure.get("mandatory_weight_bytes") != 32666638208
    ):
        raise ValueError("candidate measure is not exact4 K64 ROOT=1 B1")
    rows = measure.get("per_task")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("candidate measure lacks four task rows")
    task_wall_ms: dict[str, float] = {}
    retained: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("instance_id") not in task_ids:
            raise ValueError("candidate per-task identity drifted")
        task_id = row["instance_id"]
        if task_id in task_wall_ms:
            raise ValueError("candidate per-task identity is duplicated")
        wall_seconds = _number(row.get("wall_seconds"), f"{task_id} wall seconds")
        wall_steps = _integer(row.get("wall_steps"), f"{task_id} wall steps")
        fwd_steps = _integer(row.get("fwd_gpu_steps"), f"{task_id} forward steps")
        drafts = _integer(row.get("drafts"), f"{task_id} drafts")
        tok_per_draft = _number(
            row.get("tok_per_draft"), f"{task_id} tokens per draft"
        )
        if (
            wall_seconds <= 0
            or wall_steps < MIN_WALL_STEPS_PER_TASK
            or fwd_steps < wall_steps
            or drafts < wall_steps
            or not math.isclose(tok_per_draft, 31.0, abs_tol=1e-9)
        ):
            raise ValueError(f"{task_id} timing counters are incomplete")
        fraction = wall_steps / fwd_steps
        if fraction < MIN_RETAINED_WALL_FRACTION or fraction > 1.0:
            raise ValueError(f"{task_id} retained wall fraction drifted")
        task_wall_ms[task_id] = wall_seconds / wall_steps * 1000.0
        retained[task_id] = fraction
    if sorted(task_wall_ms) != task_ids:
        raise ValueError("candidate exact4 task set drifted")
    if (
        engagement.get("schema") != ENGAGEMENT_SCHEMA
        or engagement.get("status") != "ENGAGED"
        or engagement.get("runtime_mode") != "FULL"
        or engagement.get("batch_size") != 1
        or engagement.get("physical_rows") != 32
        or engagement.get("arm") != ARM
        or engagement.get("num_splits") != 0
        or engagement.get("layer_count") != 16
        or engagement.get("candidate_served") is not True
        or engagement.get("fallback_allowed") is not False
        or engagement.get("candidate_so_sha256") != CANDIDATE_SHA256
        or engagement.get("candidate_so_size") != CANDIDATE_SIZE
        or engagement.get("fa2_head") != FA2_HEAD
        or engagement.get("fa2_source_closure_sha256")
        != SOURCE_CLOSURE_SHA256
        or engagement.get("source_commit") != source_commit
        or engagement.get("patch_source_sha256") != patch_source_sha256
        or engagement.get("pass_sidecar_sha256") != pass_sidecar_sha256
        or sorted(engagement.get("task_ids", [])) != task_ids
        or engagement.get("subset_sha256") != EXACT4_SUBSET_SHA256
    ):
        raise ValueError("qrow32 nosplit FULL graph engagement is incomplete")
    health_tasks = health.get("tasks")
    if (
        health.get("swe_orchestrator_rc") != 0
        or not isinstance(health_tasks, list)
        or len(health_tasks) != 4
        or sorted(task.get("instance_id") for task in health_tasks) != task_ids
        or any(task.get("codex_timed_out") is not False for task in health_tasks)
        or any(task.get("verdict") == "missing" for task in health_tasks)
    ):
        raise ValueError("health record does not prove four clean canonical tasks")
    checks = audit.get("checks")
    audit_subset = audit.get("subset")
    if (
        audit.get("schema")
        not in {
            "fr13-fixed32-chat-task-provenance-audit-v2",
            "fr13-fixed32-chat-task-provenance-audit-v3",
        }
        or audit.get("mode") != "hydra27_fixed32"
        or not isinstance(audit_subset, dict)
        or audit_subset.get("sha256") != EXACT4_SUBSET_SHA256
        or audit_subset.get("task_count") != 4
        or sorted(audit_subset.get("task_ids", [])) != task_ids
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("authenticated traffic audit is not clean exact4")
    u95 = equal_task_u95(list(task_wall_ms.values()))
    u95["cap_ms"] = cap_ms
    u95["descriptive_screen_pass"] = u95["u95_ms"] <= cap_ms
    return {
        "schema": SCHEMA,
        "status": "complete",
        "run_classification": "real_swe_verified_exact4_qrow32_nosplit",
        "task_ids": task_ids,
        "task_count": 4,
        "batch_size": 1,
        "concurrency": 1,
        "topology": "hydra27_fixed32",
        "physical_rows": 32,
        "logical_drafts": 27,
        "valid_mask": "0x7abdffff",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": block_map_sha256,
        "source_commit": source_commit,
        "patch_source_sha256": patch_source_sha256,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "live_pass_sha256": pass_sha256,
        "production_sidecar_sha256": pass_sidecar_sha256,
        "runner_sha256": runner_sha256,
        "subset_sha256": EXACT4_SUBSET_SHA256,
        "subset_file_sha256": _sha(subset_raw),
        "measure_sha256": _sha(measure_raw),
        "engagement_sha256": _sha(engagement_raw),
        "health_sha256": _sha(health_raw),
        "authenticated_traffic_audit_sha256": _sha(audit_raw),
        "qrow16_historical_measure_sha256": _sha(baseline_raw),
        "qrow16_historical_context_only": True,
        "timing_eligible": True,
        "formal_floor_acceptance_eligible": False,
        "speed_claim_scope": "this exact4 candidate campaign only",
        "per_task_wall_ms": task_wall_ms,
        "per_task_retained_wall_fraction": retained,
        "descriptive_equal_task_one_sided_u95": u95,
        "mandatory_weight_floor_ms": floor_ms,
        "exact16_eligible": bool(u95["descriptive_screen_pass"]),
        "exact16_rule": "eligible only when exact4 descriptive U95 is at or below cap",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", required=True, type=Path)
    parser.add_argument("--measure", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--engagement", required=True, type=Path)
    parser.add_argument("--health", required=True, type=Path)
    parser.add_argument("--traffic-audit", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--patch-source-sha256", required=True)
    parser.add_argument("--pass-sha256", required=True)
    parser.add_argument("--pass-sidecar-sha256", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--block-map-sha256", required=True)
    parser.add_argument("--floor-ms", required=True, type=float)
    parser.add_argument("--cap-ms", required=True, type=float)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = reduce_timing(
        subset_path=args.subset,
        measure_path=args.measure,
        baseline_path=args.baseline,
        engagement_path=args.engagement,
        health_path=args.health,
        traffic_audit_path=args.traffic_audit,
        source_commit=args.source_commit,
        patch_source_sha256=args.patch_source_sha256,
        pass_sha256=args.pass_sha256,
        pass_sidecar_sha256=args.pass_sidecar_sha256,
        runner_sha256=args.runner_sha256,
        block_map_sha256=args.block_map_sha256,
        floor_ms=args.floor_ms,
        cap_ms=args.cap_ms,
        arm=args.arm,
    )
    args.out.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
