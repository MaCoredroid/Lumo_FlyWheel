#!/usr/bin/env python3
"""Reduce one authenticated BV64 committer SWE-Verified diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


class ResultError(RuntimeError):
    """The diagnostic evidence is incomplete or internally inconsistent."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ResultError(f"cannot read {label}: {path}: {error}") from error
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise ResultError(f"{label} has invalid size: {path}")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ResultError(f"invalid {label}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ResultError(f"{label} is not a JSON object: {path}")
    return payload, raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ResultError(f"{label} must be a regular non-symlink file: {path}")


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResultError(f"{label} must be a nonnegative integer")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise ResultError(f"{label} must be {qualifier}")
    return float(value)


def _int_map(value: Any, keys: set[str], label: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ResultError(f"{label} keys do not match {sorted(keys)}")
    return {key: _int(item, f"{label}.{key}") for key, item in value.items()}


def _read_env(path: Path) -> dict[str, str]:
    _regular_file(path, "container environment")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in result:
            raise ResultError(f"container environment repeats {key}")
        result[key] = value
    return result


def _snapshot_from_ref(
    arm_dir: Path,
    reference: Any,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(reference, dict):
        raise ResultError(f"{label} snapshot reference is missing")
    path_text = reference.get("path")
    digest = reference.get("sha256")
    if not isinstance(path_text, str) or not re.fullmatch(
        r"[0-9a-f]{64}", str(digest)
    ):
        raise ResultError(f"{label} snapshot reference is malformed")
    path = Path(path_text)
    try:
        resolved = path.resolve(strict=True)
        arm_resolved = arm_dir.resolve(strict=True)
    except OSError as error:
        raise ResultError(f"{label} snapshot path is unavailable: {error}") from error
    if not resolved.is_relative_to(arm_resolved) or path.is_symlink():
        raise ResultError(f"{label} snapshot escapes the arm directory")
    payload, raw = _load_json(resolved, f"{label} runtime snapshot")
    if _sha256(raw) != digest:
        raise ResultError(f"{label} runtime snapshot SHA-256 mismatch")
    return payload, raw


def _qualification_artifact(
    arm_dir: Path,
    batch_size: int,
    expected_tasks: list[str],
) -> tuple[dict[str, Any], bytes]:
    verified = arm_dir / "swe_out" / "verified"
    if batch_size == 1:
        paths = list(verified.glob("per_task/*/fixed32_task_boundary.json"))
        if len(paths) != 1:
            raise ResultError("B1 requires exactly one fixed32 task boundary")
        payload, raw = _load_json(paths[0], "B1 CFWD qualification boundary")
        if (
            payload.get("schema") != "fr13-fixed32-task-boundary-v1"
            or payload.get("run_classification")
            != "cfwd_layer_batch_real_swe_qualification"
            or payload.get("instance_id") != expected_tasks[0]
        ):
            raise ResultError("B1 CFWD qualification identity drifted")
        return payload, raw
    path = verified / "fixed32_cfwd_b4_qualification_campaign.json"
    payload, raw = _load_json(path, "B4 CFWD qualification campaign")
    if (
        payload.get("schema")
        != "fr13-fixed32-cfwd-b4-qualification-campaign-v1"
        or payload.get("run_classification")
        != "cfwd_layer_batch_real_swe_b4_qualification"
        or payload.get("batch_size") != 4
        or payload.get("concurrency") != 4
        or payload.get("task_count") != 4
        or payload.get("task_ids") != expected_tasks
        or payload.get("action_succeeded") is not True
        or payload.get("state")
        not in {"coverage_incomplete", "qualified_process_local"}
    ):
        raise ResultError("B4 CFWD qualification identity drifted")
    return payload, raw


def _validate_snapshots(
    *,
    arm_dir: Path,
    qualification: dict[str, Any],
    batch_size: int,
    expected_events: int,
) -> tuple[int, int, dict[str, int]]:
    pre, _ = _snapshot_from_ref(
        arm_dir, qualification.get("pre_runtime_snapshot"), "pre"
    )
    post, _ = _snapshot_from_ref(
        arm_dir, qualification.get("post_runtime_snapshot"), "post"
    )
    if (
        pre.get("mode") != "hydra27_fixed32"
        or post.get("mode") != "hydra27_fixed32"
    ):
        raise ResultError("runtime snapshot mode drifted")
    try:
        pre_metrics = pre["metrics"]
        post_metrics = post["metrics"]
        pre_committer = pre_metrics["committer"]
        post_committer = post_metrics["committer"]
    except (KeyError, TypeError) as error:
        raise ResultError("runtime snapshot lacks committer metrics") from error
    if not all(
        isinstance(item, dict)
        for item in (pre_metrics, post_metrics, pre_committer, post_committer)
    ):
        raise ResultError("runtime snapshot metrics are malformed")

    batch_keys = {str(batch) for batch in range(1, 5)}
    ready_keys = {str(batch) for batch in range(1, batch_size + 1)}
    pre_replays = _int(
        pre_committer.get("actual_replays_enqueued"), "pre actual replays"
    )
    post_replays = _int(
        post_committer.get("actual_replays_enqueued"), "post actual replays"
    )
    if post_replays - pre_replays != expected_events:
        raise ResultError("committer replay delta does not equal complete events")
    pre_by_batch = _int_map(
        pre_committer.get("actual_replays_by_batch"),
        batch_keys,
        "pre replays by batch",
    )
    post_by_batch = _int_map(
        post_committer.get("actual_replays_by_batch"),
        batch_keys,
        "post replays by batch",
    )
    replay_delta = {
        key: post_by_batch[key] - pre_by_batch[key] for key in batch_keys
    }
    if any(value < 0 for value in replay_delta.values()) or sum(
        replay_delta.values()
    ) != expected_events:
        raise ResultError("committer replay histogram does not reconcile")
    if any(replay_delta[str(batch)] for batch in range(batch_size + 1, 5)):
        raise ResultError("committer replay exceeded the server capacity")

    for side, committer in (("pre", pre_committer), ("post", post_committer)):
        if (
            committer.get("all_batches_ready") is not True
            or committer.get("fast_route_ready") is not True
            or _int(committer.get("captures"), f"{side} captures") != batch_size
            or _int(committer.get("preseeded_graphs"), f"{side} preseeded graphs")
            != batch_size
            or committer.get("preseeded_batches")
            != list(range(1, batch_size + 1))
        ):
            raise ResultError("committer capture/preseed engagement drifted")

    pre_nonpure = _int(
        pre_committer.get("nonpure_committer_replays_enqueued"),
        "pre nonpure replays",
    )
    post_nonpure = _int(
        post_committer.get("nonpure_committer_replays_enqueued"),
        "post nonpure replays",
    )
    if post_nonpure != pre_nonpure:
        raise ResultError("nonpure committer fallback executed during the task")
    for side, committer in (("pre", pre_committer), ("post", post_committer)):
        fallback = _int_map(
            committer.get("metadata_fusion_fallbacks_by_batch"),
            batch_keys,
            f"{side} metadata fallback",
        )
        if any(fallback.values()):
            raise ResultError("metadata fallback counter is nonzero")

    pre_attempts = _int_map(
        pre_committer.get("layer_batch_gate_attempts_by_batch"),
        ready_keys,
        "pre layer-batch attempts",
    )
    post_attempts = _int_map(
        post_committer.get("layer_batch_gate_attempts_by_batch"),
        ready_keys,
        "post layer-batch attempts",
    )
    attempt_delta = {
        key: post_attempts[key] - pre_attempts[key] for key in ready_keys
    }
    if any(value < 0 for value in attempt_delta.values()):
        raise ResultError("layer-batch attempt counter regressed")

    for label in ("sfwd", "dfwd", "cfwd"):
        try:
            pre_span = pre_metrics[label]
            post_span = post_metrics[label]
        except KeyError as error:
            raise ResultError(f"runtime snapshot lacks {label} metrics") from error
        counter = "steps" if label == "sfwd" else "spans"
        if (
            _int(post_span.get(counter), f"post {label} {counter}")
            - _int(pre_span.get(counter), f"pre {label} {counter}")
            != expected_events
        ):
            raise ResultError(f"{label} event count does not reconcile")
    return sum(attempt_delta.values()), post_replays - pre_replays, replay_delta


def _validate_runtime(
    *,
    runtime_log: Path,
    container_env: Path,
    batch_size: int,
) -> bytes:
    _regular_file(runtime_log, "runtime log")
    raw = runtime_log.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    env = _read_env(container_env)
    expected_env = {
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
        "ENFORCE_EAGER": "0",
        "FR13_CFWD_GPU_TIMER": "1",
        "FR13_DFWD_GPU_TIMER": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_ROOT": "1",
        "FR13_FIXED32_B1_DIAGNOSTIC": "1" if batch_size == 1 else "0",
        "FR13_FIXED32_COMMITTER_BV64_WARP4": "1",
        "FR13_FIXED32_COMMITTER_DECAY_RING": "0",
        "FR13_FIXED32_COMMITTER_DIRECT_METADATA": "0",
        "FR13_FIXED32_COMMITTER_GATE_RING": "0",
        "FR13_FIXED32_COMMITTER_KNORM_RING": "0",
        "FR13_FIXED32_COMMITTER_LAYER_BATCH": "1",
        "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION": "1",
        "FR13_FIXED32_COMMITTER_METADATA_FUSION": "0",
        "FR13_FIXED32_COMMITTER_STICKY_GUARD": "0",
        "FR13_FIXED32_MODE": "hydra27_fixed32",
        "FR13_SFWD_GPU_TIMER": "1",
        "MAX_NUM_SEQS": str(batch_size),
        "SWE_CONCURRENCY": str(batch_size),
    }
    for key, value in expected_env.items():
        if env.get(key) != value:
            raise ResultError(f"container environment lacks {key}={value}")

    pattern = re.compile(
        r"\[FR13_FIXED32_COMMIT_DEVICE_FILL\] preseeded: "
        r"mode=hydra27_fixed32 B=(\d+) .*"
        r"layer_batch=1 bv64_warp4=1 replays=1"
    )
    captured_batches = sorted(int(value) for value in pattern.findall(text))
    if captured_batches != list(range(1, batch_size + 1)):
        raise ResultError("BV64 committer capture markers do not match capacity")
    if (
        "[FR13_FIXED32_COMMIT_DEVICE_FILL ENGAGED] "
        "mode=hydra27_fixed32" not in text
        or "Graph capturing finished" not in text
    ):
        raise ResultError("BV64 committer or FULL CUDA graph did not engage")
    if "committer layer-batch byte gate failed" in text:
        raise ResultError("BV64 committer byte gate failed")
    return raw


def reduce_result(
    *,
    arm_dir: Path,
    batch_size: int,
    subset_path: Path,
    measurement_path: Path,
    runtime_log: Path,
    container_env: Path,
    source_commit: str,
    runner_sha256: str,
) -> dict[str, Any]:
    if batch_size not in (1, 4):
        raise ResultError("batch size must be exactly 1 or 4")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ResultError("source commit must be a full Git object ID")
    if not re.fullmatch(r"[0-9a-f]{64}", runner_sha256):
        raise ResultError("runner SHA-256 is malformed")
    subset, subset_raw = _load_json(subset_path, "SWE-Verified subset")
    expected_tasks = subset.get("instance_ids")
    if (
        subset.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or not isinstance(expected_tasks, list)
        or len(expected_tasks) != batch_size
        or any(not isinstance(task, str) or not task for task in expected_tasks)
    ):
        raise ResultError("SWE-Verified subset identity drifted")

    qualification, qualification_raw = _qualification_artifact(
        arm_dir, batch_size, expected_tasks
    )
    if any(
        qualification.get(key) is not False
        for key in (
            "acceptance_valid",
            "performance_measurement",
            "timing_eligible",
            "floor_acceptance_eligible",
        )
    ):
        raise ResultError("qualification was mislabeled as performance evidence")
    interval = qualification.get("forward_step_interval")
    if not isinstance(interval, dict):
        raise ResultError("qualification has no complete forward interval")
    start = _int(interval.get("start_forward_step"), "interval start")
    end = _int(interval.get("end_forward_step"), "interval end")
    expected_events = _int(
        interval.get("expected_complete_events"), "complete events"
    )
    if expected_events <= 0 or end - start != expected_events:
        raise ResultError("qualification forward interval does not reconcile")

    shadow_replays, total_replays, replay_delta = _validate_snapshots(
        arm_dir=arm_dir,
        qualification=qualification,
        batch_size=batch_size,
        expected_events=expected_events,
    )
    coverage = qualification.get("qualification_coverage")
    if not isinstance(coverage, dict):
        raise ResultError("qualification coverage artifact is missing")
    artifact_attempts = _int_map(
        coverage.get("attempt_delta_by_batch"),
        {str(batch) for batch in range(1, batch_size + 1)},
        "artifact attempt delta",
    )
    if sum(artifact_attempts.values()) != shadow_replays:
        raise ResultError("qualification attempts differ from runtime counters")
    candidate_replays = total_replays - shadow_replays
    if candidate_replays <= 0:
        raise ResultError("no BV64 candidate-served replay was observed")

    runtime_raw = _validate_runtime(
        runtime_log=runtime_log,
        container_env=container_env,
        batch_size=batch_size,
    )
    measurement, measurement_raw = _load_json(
        measurement_path, "deploy-speed measurement"
    )
    if (
        measurement.get("schema") != "fr13.measure.deploy_speed.v1"
        or measurement.get("regime") != "deployment"
        or measurement.get("batch_size") != batch_size
        or measurement.get("n_tasks") != batch_size
        or measurement.get("task_instance_ids") != expected_tasks
        or measurement.get("draft_vocab_k") != 65536
        or measurement.get("draft_vocab_root") != 1
        or measurement.get("draft_head_fp8") is not False
        or not isinstance(measurement.get("engagement"), dict)
        or measurement["engagement"].get("engaged") is not True
        or measurement.get("mandatory_weight_bytes") != 25210209416
        or measurement.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise ResultError("deploy-speed measurement binding drifted")
    weight_floor_ms = _number(
        measurement.get("weight_floor_ms"), "weight floor", positive=True
    )
    if not math.isclose(
        weight_floor_ms, 92.345089436, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ResultError("deploy-speed weight floor drifted")

    step_wall_ms = _number(
        measurement.get("step_wall_ms"), "step wall", positive=True
    )
    sfwd_ms = 1000.0 * _number(
        measurement.get("s_per_fwd_gpu_per_forward"),
        "SFWD seconds per forward",
        positive=True,
    )
    dfwd_ms = _number(
        measurement.get("drafter_gpu_ms_per_step"), "DFWD ms", positive=True
    )
    cfwd_ms = _number(
        measurement.get("committer_gpu_ms_per_step"), "CFWD ms", positive=True
    )
    events_per_step = _number(
        measurement.get("events_per_step"), "events per step", positive=True
    )
    overhead_event_ms = _number(
        measurement.get("overhead_other_ms_per_event"), "other overhead"
    )
    other_ms = overhead_event_ms * events_per_step
    if not math.isclose(
        sfwd_ms + dfwd_ms + cfwd_ms + other_ms,
        step_wall_ms,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ResultError("wall/SFWD/DFWD/CFWD/other breakdown does not sum")
    for key in (
        "accept_per_event",
        "committed_per_event",
        "derived_tps_fullstep_gpu",
        "floor_ms",
        "floor_ratio",
        "measured_tps_fullstep_wall",
        "wall_s_per_event",
    ):
        _number(measurement.get(key), key, positive=True)

    return {
        "schema": "fr13.fixed32.committer_bv64_warp4.real_diagnostic.v1",
        "status": "PASS",
        "suite": "SWE-Verified",
        "run_classification": (
            "one_real_swe_verified_b1_committer_bv64_diagnostic"
            if batch_size == 1
            else "real_swe_verified_exact4_b4_committer_bv64_diagnostic"
        ),
        "batch_size": batch_size,
        "task_ids": expected_tasks,
        "mode": "hydra27_fixed32",
        "physical_rows": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "value_tile": 64,
        "kernel_warps": 4,
        "layer_batch": True,
        "capture_engaged": True,
        "candidate_served": True,
        "candidate_served_replays": candidate_replays,
        "qualification_reference_served_replays": shadow_replays,
        "fallback_replays": 0,
        "total_committer_replays": total_replays,
        "committer_replay_delta_by_batch": replay_delta,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "measurement_scope": "informative_real_task_diagnostic",
        "phase_breakdown_ms_per_step": {
            "wall": step_wall_ms,
            "sfwd": sfwd_ms,
            "dfwd": dfwd_ms,
            "cfwd": cfwd_ms,
            "other": other_ms,
        },
        "accept_per_event": measurement["accept_per_event"],
        "committed_per_event": measurement["committed_per_event"],
        "fullstep_wall_tps": measurement["measured_tps_fullstep_wall"],
        "fullstep_gpu_component_tps": measurement["derived_tps_fullstep_gpu"],
        "floor_ms": measurement["floor_ms"],
        "floor_ratio": measurement["floor_ratio"],
        "mandatory_weight_bytes": measurement["mandatory_weight_bytes"],
        "weight_floor_ms": weight_floor_ms,
        "source_commit": source_commit,
        "runner_sha256": runner_sha256,
        "subset_sha256": _sha256(subset_raw),
        "qualification_sha256": _sha256(qualification_raw),
        "runtime_log_sha256": _sha256(runtime_raw),
        "measurement_sha256": _sha256(measurement_raw),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--runtime-log", type=Path, required=True)
    parser.add_argument("--container-env", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = reduce_result(
        arm_dir=args.arm_dir,
        batch_size=args.batch_size,
        subset_path=args.subset,
        measurement_path=args.measurement,
        runtime_log=args.runtime_log,
        container_env=args.container_env,
        source_commit=args.source_commit,
        runner_sha256=args.runner_sha256,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
