#!/usr/bin/env python3
"""Reduce one real exact4 fixed32 stock/overlap timing pair.

The candidate census is cumulative, but every record is bound to the existing
fixed32 flush generation.  The reducer therefore brackets only the canonical
SWE-Verified task interval and excludes boot/warmup work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.k64.commit_draft_overlap.exact4_pair.v1"
OVERLAP_SCHEMA = "fr13.fixed32.commit_draft_overlap.v1"
OVERLAP_ARM = "fr13.fixed32.k64.commit_draft_overlap.v1"
BOUNDARY_SCHEMA = "fr13-fixed32-task-boundary-v1"
SPEED_SCHEMA = "fr13.measure.deploy_speed.v1"
SUBSET_PATH = Path("config/fr13_fixed32/subset_b4_four.json")
SUBSET_SHA256 = "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
FLOOR_MS = 119.658015414
U95_CAP_MS = 137.6067177261
MODES = {
    "tail6_fixed32": {"topology": "Tail23", "logical_drafts": 23},
    "hydra27_fixed32": {"topology": "Hydra27", "logical_drafts": 27},
}


class GateError(RuntimeError):
    pass


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file() or path.is_symlink():
        raise GateError(f"required regular JSON is missing: {path}")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid ASCII JSON {path}: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"JSON root is not an object: {path}")
    return payload, hashlib.sha256(raw).hexdigest()


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GateError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise GateError(f"{label} is outside its finite range")
    return result


def _canonical_tasks(repo: Path) -> tuple[str, ...]:
    subset = repo / SUBSET_PATH
    raw = subset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SUBSET_SHA256:
        raise GateError("canonical exact4 subset SHA-256 drifted")
    payload = json.loads(raw.decode("ascii"))
    tasks = payload.get("instance_ids") or payload.get("task_ids")
    if not isinstance(tasks, list) or len(tasks) != 4:
        raise GateError("canonical exact4 subset does not contain four tasks")
    normalized = tuple(str(task) for task in tasks)
    if len(set(normalized)) != 4 or any(not task for task in normalized):
        raise GateError("canonical exact4 task identities are malformed")
    return normalized


def _task_boundaries(arm: Path, expected_tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    paths = sorted(arm.glob("swe_out/*/per_task/*/fixed32_task_boundary.json"))
    if len(paths) != 4:
        raise GateError(f"{arm} has {len(paths)} task boundaries, expected 4")
    records = []
    for path in paths:
        payload, sha256 = _read_json(path)
        if payload.get("schema") != BOUNDARY_SCHEMA:
            raise GateError(f"task boundary schema drifted: {path}")
        task = payload.get("instance_id")
        pre = payload.get("pre")
        post = payload.get("post")
        if (
            not isinstance(task, str)
            or not isinstance(pre, dict)
            or not isinstance(post, dict)
            or pre.get("action") != "snapshot"
            or post.get("action") != "snapshot"
        ):
            raise GateError(f"task boundary is incomplete: {path}")
        pre_generation = _exact_int(pre.get("generation"), f"{path}:pre generation")
        post_generation = _exact_int(post.get("generation"), f"{path}:post generation")
        if post_generation <= pre_generation:
            raise GateError(f"task boundary generations are unordered: {path}")
        records.append(
            {
                "task": task,
                "pre_generation": pre_generation,
                "post_generation": post_generation,
                "producer_pid": _exact_int(
                    payload.get("producer_pid"), f"{path}:producer_pid"
                ),
                "sha256": sha256,
            }
        )
    if set(record["task"] for record in records) != set(expected_tasks):
        raise GateError(f"{arm} task identities differ from canonical exact4")
    return records


def _overlap_snapshot(arm: Path, generation: int) -> dict[str, Any]:
    path = arm / "logs" / f"fr13_fixed32_commit_draft_overlap.{generation}.json"
    payload, sha256 = _read_json(path)
    required = {
        "schema",
        "armed",
        "arm_contract",
        "begun",
        "sealed",
        "fenced",
        "flush_fenced",
        "timed_spans",
        "tail_gpu_ms_total",
        "tail_gpu_ms_per_span",
        "pending",
        "order_reconciled",
        "by_batch",
        "event_slots",
        "streams",
        "generation",
        "action",
        "producer_pid",
    }
    if set(payload) != required:
        raise GateError(f"overlap snapshot keys drifted: {path}")
    if (
        payload["schema"] != OVERLAP_SCHEMA
        or payload["armed"] is not True
        or payload["arm_contract"] != OVERLAP_ARM
        or payload["pending"] is not False
        or payload["order_reconciled"] is not True
        or payload["event_slots"] != 2
        or payload["streams"] != 1
        or payload["generation"] != generation
        or payload["action"] not in ("snapshot", "final")
    ):
        raise GateError(f"overlap lifecycle attestation failed: {path}")
    begun = _exact_int(payload["begun"], f"{path}:begun")
    sealed = _exact_int(payload["sealed"], f"{path}:sealed")
    fences = _exact_int(payload["fenced"], f"{path}:fenced") + _exact_int(
        payload["flush_fenced"], f"{path}:flush_fenced"
    )
    spans = _exact_int(payload["timed_spans"], f"{path}:timed_spans")
    if begun != sealed or begun != fences or begun != spans:
        raise GateError(f"overlap counters do not reconcile: {path}")
    by_batch = payload["by_batch"]
    if not isinstance(by_batch, dict) or set(by_batch) != {"1", "2", "3", "4"}:
        raise GateError(f"overlap batch histogram drifted: {path}")
    normalized_batch = {
        key: _exact_int(value, f"{path}:by_batch.{key}")
        for key, value in by_batch.items()
    }
    if sum(normalized_batch.values()) != begun:
        raise GateError(f"overlap batch histogram does not sum to spans: {path}")
    payload["by_batch"] = normalized_batch
    payload["tail_gpu_ms_total"] = _finite(
        payload["tail_gpu_ms_total"], f"{path}:tail_gpu_ms_total"
    )
    payload["sha256"] = sha256
    return payload


def _subtract_snapshot(start: dict[str, Any], end: dict[str, Any], batch: int) -> dict[str, Any]:
    if end["producer_pid"] != start["producer_pid"]:
        raise GateError("overlap snapshots cross EngineCore producers")
    counters = {}
    for key in ("begun", "sealed", "fenced", "flush_fenced", "timed_spans"):
        counters[key] = int(end[key]) - int(start[key])
        if counters[key] < 0:
            raise GateError(f"overlap counter regressed across exact4: {key}")
    fences = counters["fenced"] + counters["flush_fenced"]
    if not (
        counters["begun"]
        == counters["sealed"]
        == counters["timed_spans"]
        == fences
        > 0
    ):
        raise GateError("exact4 overlap interval lifecycle does not reconcile")
    by_batch = {
        key: end["by_batch"][key] - start["by_batch"][key]
        for key in ("1", "2", "3", "4")
    }
    if any(value < 0 for value in by_batch.values()) or sum(by_batch.values()) != counters["begun"]:
        raise GateError("exact4 overlap batch histogram does not reconcile")
    if batch == 1 and any(by_batch[key] for key in ("2", "3", "4")):
        raise GateError("B1 overlap interval contains a multi-request event")
    request_events = sum(int(key) * value for key, value in by_batch.items())
    tail_ms = float(end["tail_gpu_ms_total"]) - float(start["tail_gpu_ms_total"])
    if not math.isfinite(tail_ms) or tail_ms <= 0.0 or request_events <= 0:
        raise GateError("exact4 overlap timing interval is empty")
    return {
        **counters,
        "by_batch": by_batch,
        "request_events": request_events,
        "tail_gpu_ms_total": tail_ms,
        "tail_gpu_ms_per_physical_step": tail_ms / counters["begun"],
        "tail_gpu_ms_per_request_event": tail_ms / request_events,
        "start_generation": start["generation"],
        "end_generation": end["generation"],
        "start_sha256": start["sha256"],
        "end_sha256": end["sha256"],
    }


def _speed(path: Path, tasks: tuple[str, ...], batch: int) -> tuple[dict[str, Any], str]:
    payload, sha256 = _read_json(path)
    if (
        payload.get("schema") != SPEED_SCHEMA
        or payload.get("n_tasks") != 4
        or payload.get("batch_size") != batch
        or set(payload.get("task_instance_ids", ())) != set(tasks)
        or payload.get("engagement", {}).get("engaged") is not True
        or float(payload.get("engagement", {}).get("tok_per_draft", -1)) != 31.0
    ):
        raise GateError(f"deploy-speed identity or engagement failed: {path}")
    for key in (
        "wall_s_per_event",
        "measured_tps_fullstep_wall",
        "s_per_fwd_gpu",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "events_per_step",
        "accept_per_event",
        "committed_per_event",
    ):
        payload[key] = _finite(payload.get(key), f"{path}:{key}", positive=True)
    return payload, sha256


def _breakdown(speed: dict[str, Any], tail: dict[str, Any] | None) -> dict[str, Any]:
    occupancy = speed["events_per_step"]
    wall_ms = speed["wall_s_per_event"] * 1000.0
    sfwd_ms = speed["s_per_fwd_gpu"] * 1000.0
    dfwd_ms = speed["drafter_gpu_ms_per_step"] / occupancy
    cfwd_ms = speed["committer_gpu_ms_per_step"] / occupancy
    residual_ms = wall_ms - sfwd_ms - dfwd_ms - cfwd_ms
    result = {
        "full_step_wall_ms_per_request_event": wall_ms,
        "full_wall_tps": speed["measured_tps_fullstep_wall"],
        "accepted_drafts_per_request_event": speed["accept_per_event"],
        "committed_tokens_per_request_event": speed["committed_per_event"],
        "events_per_physical_step": occupancy,
        "sfwd_gpu_ms_per_request_event": sfwd_ms,
        "dfwd_gpu_ms_per_request_event": dfwd_ms,
        "cfwd_default_stream_gpu_ms_per_request_event": cfwd_ms,
        "residual_wall_ms_per_request_event": residual_ms,
        "wall_over_floor_ratio": wall_ms / FLOOR_MS,
        "wall_gap_to_u95_cap_ms": wall_ms - U95_CAP_MS,
        "component_sum_valid": tail is None,
    }
    if tail is not None:
        result.update(
            commit_tail_gpu_ms_per_physical_step=tail[
                "tail_gpu_ms_per_physical_step"
            ],
            commit_tail_gpu_ms_per_request_event=tail[
                "tail_gpu_ms_per_request_event"
            ],
            component_sum_valid=False,
            component_sum_note=(
                "commit tail overlaps the default-stream DFWD interval; do not "
                "sum it with SFWD/DFWD/CFWD"
            ),
        )
    return result


def reduce_pair(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    stock_arm = args.stock_arm.resolve()
    candidate_arm = args.candidate_arm.resolve()
    if args.mode not in MODES or args.batch not in (1, 4):
        raise GateError("mode must be fixed32 Tail/Hydra and batch must be 1 or 4")
    tasks = _canonical_tasks(repo)
    stock_boundaries = _task_boundaries(stock_arm, tasks)
    candidate_boundaries = _task_boundaries(candidate_arm, tasks)
    if set(record["task"] for record in stock_boundaries) != set(
        record["task"] for record in candidate_boundaries
    ):
        raise GateError("stock and candidate task sets differ")

    stock_arm_path = stock_arm / "logs/fr13_fixed32_commit_draft_overlap.arm"
    if stock_arm_path.exists() or list(
        stock_arm.glob("logs/fr13_fixed32_commit_draft_overlap.*.json")
    ):
        raise GateError("stock arm contains overlap state")
    candidate_arm_path = candidate_arm / "logs/fr13_fixed32_commit_draft_overlap.arm"
    if (
        not candidate_arm_path.is_file()
        or candidate_arm_path.is_symlink()
        or candidate_arm_path.read_text(encoding="ascii").strip() != OVERLAP_ARM
    ):
        raise GateError("candidate overlap arm attestation is absent or malformed")

    earliest = min(candidate_boundaries, key=lambda row: row["pre_generation"])
    latest = max(candidate_boundaries, key=lambda row: row["post_generation"])
    start = _overlap_snapshot(candidate_arm, earliest["pre_generation"])
    end = _overlap_snapshot(candidate_arm, latest["post_generation"])
    if start["producer_pid"] != earliest["producer_pid"] or end["producer_pid"] != latest["producer_pid"]:
        raise GateError("candidate overlap census producer differs from task bracket")
    interval = _subtract_snapshot(start, end, args.batch)

    # Every task must be covered by a positive, reconciled cumulative interval.
    task_intervals = {}
    for boundary in candidate_boundaries:
        pre = _overlap_snapshot(candidate_arm, boundary["pre_generation"])
        post = _overlap_snapshot(candidate_arm, boundary["post_generation"])
        task_intervals[boundary["task"]] = _subtract_snapshot(pre, post, args.batch)

    stock_speed, stock_speed_sha = _speed(args.stock_speed, tasks, args.batch)
    candidate_speed, candidate_speed_sha = _speed(
        args.candidate_speed, tasks, args.batch
    )
    stock = _breakdown(stock_speed, None)
    candidate = _breakdown(candidate_speed, interval)
    wall_gain_ms = (
        stock["full_step_wall_ms_per_request_event"]
        - candidate["full_step_wall_ms_per_request_event"]
    )
    tps_ratio = candidate["full_wall_tps"] / stock["full_wall_tps"]
    verdict = (
        "reject_wall_regression"
        if wall_gain_ms <= 0.0
        else "screen_pass_pending_repeat_ci_and_byte_gate"
    )
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "verdict": verdict,
        "run_classification": "real_swe_verified_exact4_k64_overlap_pair",
        "formal_floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "gpu_used_by_reducer": False,
        "source_commit": args.source_commit,
        "mode": args.mode,
        "logical_topology": MODES[args.mode]["topology"],
        "logical_drafts": MODES[args.mode]["logical_drafts"],
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "batch_size": args.batch,
        "concurrency": args.batch,
        "task_count": 4,
        "task_ids": list(tasks),
        "subset_sha256": SUBSET_SHA256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "mandatory_weight_floor_ms": FLOOR_MS,
        "one_sided_u95_cap_ms": U95_CAP_MS,
        "only_arm_delta": "FR13_FIXED32_COMMIT_DRAFT_OVERLAP_0_to_1",
        "stock": stock,
        "candidate": candidate,
        "candidate_overlap_exact4_interval": interval,
        "candidate_overlap_task_intervals": task_intervals,
        "candidate_minus_stock_wall_ms": -wall_gain_ms,
        "candidate_wall_saved_ms": wall_gain_ms,
        "candidate_to_stock_full_wall_tps_ratio": tps_ratio,
        "stock_speed_sha256": stock_speed_sha,
        "candidate_speed_sha256": candidate_speed_sha,
        "stock_task_boundary_sha256": {
            row["task"]: row["sha256"] for row in stock_boundaries
        },
        "candidate_task_boundary_sha256": {
            row["task"]: row["sha256"] for row in candidate_boundaries
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    os.replace(temporary, args.output)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--mode", required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--stock-arm", type=Path, required=True)
    parser.add_argument("--candidate-arm", type=Path, required=True)
    parser.add_argument("--stock-speed", type=Path, required=True)
    parser.add_argument("--candidate-speed", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = reduce_pair(args)
    except GateError as error:
        raise SystemExit(f"FAIL: {error}") from error
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
