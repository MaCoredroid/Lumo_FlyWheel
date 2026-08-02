#!/usr/bin/env python3
"""Issue and verify the fixed32 deployed-format BF16 M32 head credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.draft_head_m32_live_ab.v1"
SIDECAR_SCHEMA = "fr13.fixed32.draft_head_m32_production_pass.v2"
ENGAGEMENT_SCHEMA = "fr13.fixed32.draft_head_m32_production_engagement.v1"
EXPECTED_INSTANCE = "astropy__astropy-12907"
EXPECTED_MODE = "hydra27_fixed32"
EXPECTED_DATASET = "princeton-nlp/SWE-bench_Verified"
EXPECTED_B1_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
EXPECTED_DATASET_RECORD_SHA256 = (
    "bcadb9e2ee9c01d1951516eeb31a5864abb90adc8e3bedc5419ce4eb414517db"
)
HEX = frozenset("0123456789abcdef")
EXPECTED_GEOMETRY = {
    "batch_size": 1,
    "calls_per_event": 5,
    "input_shape": [1, 5120],
    "input_stride": [5120, 1],
    "weight_shape": [65536, 5120],
    "weight_stride": [5120, 1],
    "weight_transpose_stride": [1, 5120],
    "output_shape": [1, 65536],
    "output_stride": [65536, 1],
    "dtype": "torch.bfloat16",
}
EXPECTED_CANDIDATE = {
    "method": "UnquantizedEmbeddingMethod",
    "operation": "replicate hidden row to M32 then torch.mm with weight.t",
    "gemm_mnk": [32, 65536, 5120],
    "candidate_input_stride": [5120, 1],
    "candidate_weight_transpose_stride": [1, 5120],
    "candidate_output_stride": [65536, 1],
    "served_rows": 1,
}
EXPECTED_GRAPH_SIGNATURE = "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
LIVE_KEYS = frozenset(
    {
        "batch_size",
        "boundary_snapshot_sha256",
        "candidate",
        "candidate_source_sha256",
        "complete_work_census_events",
        "completed_events",
        "concurrency",
        "events_sha256",
        "finalized_by_fixed32_flush",
        "flush_action",
        "flush_generation",
        "flush_nonce",
        "full_logit_comparisons",
        "geometry",
        "instance_id",
        "performance_measurement",
        "producer_pid",
        "raw_bf16_mismatches",
        "schema",
        "served_return",
        "source_commit",
        "status",
        "suite",
        "task_marker",
        "work_census_last_event_index",
    }
)
ENGAGEMENT_KEYS = frozenset(
    {
        "candidate",
        "candidate_source_sha256",
        "capture_origin",
        "captured_loop_calls",
        "drafter_graph_id",
        "drafter_graph_signature",
        "execution_basis",
        "fallback_calls",
        "forward_step_index",
        "geometry",
        "observed_measured_replays_at_least",
        "production_pass_sidecar_sha256",
        "runtime_mode",
        "schema",
        "selected_root_calls",
        "source_commit",
        "status",
    }
)
RESULT_KEYS = frozenset({"schema", "ack"})
ACK_KEYS = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "generation",
        "nonce",
        "action",
        "status",
        "counters",
    }
)
COUNTER_KEYS = frozenset(
    {
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "work_census_first_forward_step",
        "work_census_last_forward_step",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    }
)
BOUNDARY_KEYS = frozenset(
    {
        "schema",
        "mode",
        "producer_pid",
        "generation",
        "nonce",
        "action",
        "counters",
        "metrics",
    }
)
BOUNDARY_METRIC_KEYS = frozenset(
    {"fixed32", "sfwd", "dfwd", "cfwd", "boot_warm", "committer", "conv_pregather"}
)
FIXED32_METRIC_KEYS = frozenset(
    {
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "complete_spec_rows",
        "spec_drafts",
        "spec_tokens",
        "batch_histogram",
        "first_forward_step",
        "last_forward_step",
        "events_sha256",
    }
)
SFWD_KEYS = frozenset(
    {
        "gpu_seconds",
        "steps",
        "drafts",
        "wall_seconds",
        "wall_drafts",
        "wall_steps",
        "wall_rejected",
    }
)
SPAN_KEYS = frozenset({"gpu_seconds", "spans"})
TRAFFIC_AUDIT_KEYS = frozenset(
    {
        "schema",
        "mode",
        "dataset_name",
        "subset",
        "checks",
        "offload_fetch_status",
        "proxy_runtime",
        "complete_stream",
        "ingress",
        "tasks",
    }
)
TRAFFIC_CHECK_KEYS = frozenset(
    {
        "all_canonical_tasks_validated",
        "all_task_identity_and_dataset_hashes_exact",
        "all_task_agent_and_eval_terminal",
        "all_trace_request_counts_match_authenticated_proxy",
        "all_proxy_attempts_match_engine_requests",
        "all_successful_engine_requests_match_census",
        "all_census_requests_inside_task_brackets",
        "no_campaign_rejections_or_aborted_requests",
        "no_fixed32_traffic_outside_task_brackets",
        "raw_proxy_request_and_response_dumps_disabled",
    }
)
INGRESS_KEYS = frozenset(
    {
        "canonical_task_set_sha256",
        "census",
        "engine",
        "exact_proxy_engine_attempt_parity",
        "preflight",
        "proxy",
        "zero_campaign_rejections",
        "zero_failed_or_aborted_requests",
    }
)
CENSUS_KEYS = frozenset(
    {
        "all_census_requests_authenticated",
        "all_census_requests_inside_task_brackets",
        "all_successful_requests_present",
        "bytes",
        "event_count",
        "event_schema",
        "path",
        "per_task_request_step_memberships",
        "request_step_memberships",
        "sha256",
        "successful_engine_requests",
        "terminal_schema",
    }
)


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_regular_file(path: Path, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload, raw


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase 40-character commit")
    return value


def _require_exact_keys(
    value: Any, expected: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != expected:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(
            f"{label} key set drifted: expected={sorted(expected)} actual={actual}"
        )
    return value


def _require_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    value = _require_nonnegative_int(value, label)
    if value == 0:
        raise ValueError(f"{label} must be positive")
    return value


def _require_nonnegative_float(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _validate_artifact_identity(value: Any, label: str) -> dict[str, Any]:
    identity = _require_exact_keys(
        value, frozenset({"path", "sha256", "bytes"}), label
    )
    if not isinstance(identity["path"], str) or not identity["path"]:
        raise ValueError(f"{label}.path must be nonempty")
    _require_sha256(identity["sha256"], f"{label}.sha256")
    _require_positive_int(identity["bytes"], f"{label}.bytes")
    return identity


def _validate_terminal_counters(
    counters: Any, *, expected_events: int, label: str
) -> dict[str, Any]:
    counters = _require_exact_keys(counters, COUNTER_KEYS, label)
    for key in (
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "sfwd_pending",
        "dfwd_pending",
        "cfwd_pending",
    ):
        _require_nonnegative_int(counters[key], f"{label}.{key}")
    for key in (
        "work_census_first_forward_step",
        "work_census_last_forward_step",
    ):
        _require_nonnegative_int(counters[key], f"{label}.{key}")
    expected = {
        "pure_decode_forward_steps": expected_events,
        "complete_work_census_events": expected_events,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": expected_events - 1,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    if counters != expected:
        raise ValueError(f"{label} is not the exact closed B1 event census")
    return counters


def validate_live_result(
    payload: dict[str, Any],
    *,
    expected_source_sha256: str,
) -> dict[str, str | int]:
    expected_source_sha256 = _require_sha256(
        expected_source_sha256, "qualified candidate source"
    )
    if set(payload) != LIVE_KEYS:
        raise ValueError("draft-head M32 live result key set drifted")
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("draft-head M32 live result is not a PASS record")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("candidate_source_sha256") != expected_source_sha256
        or payload.get("served_return") != "reference BF16 logits unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("finalized_by_fixed32_flush") is not True
        or payload.get("flush_action") != "final"
    ):
        raise ValueError("draft-head M32 live result provenance drifted")
    _require_commit(payload.get("source_commit"), "live source")
    _require_sha256(payload.get("events_sha256"), "live event census")
    _require_sha256(
        payload.get("boundary_snapshot_sha256"), "live boundary snapshot"
    )
    _require_sha256(payload.get("flush_nonce"), "live flush nonce")
    completed_events = payload.get("completed_events")
    census_events = payload.get("complete_work_census_events")
    comparisons = payload.get("full_logit_comparisons")
    if (
        type(completed_events) is not int
        or completed_events < 1
        or type(census_events) is not int
        or census_events != completed_events
        or payload.get("work_census_last_event_index") != census_events - 1
        or type(payload.get("flush_generation")) is not int
        or payload["flush_generation"] < 1
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
        or type(comparisons) is not int
        or comparisons != completed_events * EXPECTED_GEOMETRY["calls_per_event"]
        or payload.get("raw_bf16_mismatches") != 0
    ):
        raise ValueError("draft-head M32 live comparison census drifted")
    return {
        "source_commit": payload["source_commit"],
        "completed_events": completed_events,
    }


def validate_live_evidence(
    *,
    live_payload: dict[str, Any],
    final_flush_path: Path,
    boundary_snapshot_path: Path,
) -> dict[str, str | int]:
    final_flush, _ = load_json(final_flush_path)
    boundary, boundary_raw = load_json(boundary_snapshot_path)
    _require_exact_keys(final_flush, RESULT_KEYS, "final flush result")
    ack = _require_exact_keys(final_flush["ack"], ACK_KEYS, "final flush ack")
    _require_exact_keys(boundary, BOUNDARY_KEYS, "final boundary")
    metrics = _require_exact_keys(
        boundary["metrics"], BOUNDARY_METRIC_KEYS, "final boundary metrics"
    )
    fixed32 = _require_exact_keys(
        metrics["fixed32"], FIXED32_METRIC_KEYS, "final fixed32 metrics"
    )
    sfwd = _require_exact_keys(metrics["sfwd"], SFWD_KEYS, "final SFWD metrics")
    dfwd = _require_exact_keys(metrics["dfwd"], SPAN_KEYS, "final DFWD metrics")
    cfwd = _require_exact_keys(metrics["cfwd"], SPAN_KEYS, "final CFWD metrics")
    for label in ("boot_warm", "committer", "conv_pregather"):
        if not isinstance(metrics[label], dict):
            raise ValueError(f"final {label} metrics must be an object")

    completed_events = live_payload["completed_events"]
    for record, label in ((ack, "final flush ack"), (boundary, "final boundary")):
        _require_positive_int(record["producer_pid"], f"{label}.producer_pid")
        _require_positive_int(record["generation"], f"{label}.generation")
        _require_sha256(record["nonce"], f"{label}.nonce")
    ack_counters = _validate_terminal_counters(
        ack["counters"], expected_events=completed_events, label="final flush ack"
    )
    boundary_counters = _validate_terminal_counters(
        boundary["counters"],
        expected_events=completed_events,
        label="final boundary",
    )
    for key in (
        "pure_decode_forward_steps",
        "complete_work_census_events",
        "complete_spec_rows",
        "spec_drafts",
        "spec_tokens",
    ):
        _require_nonnegative_int(fixed32[key], f"final fixed32 metrics.{key}")
    for key in ("first_forward_step", "last_forward_step"):
        _require_nonnegative_int(fixed32[key], f"final fixed32 metrics.{key}")
    _require_sha256(fixed32["events_sha256"], "final fixed32 event census")
    expected_histogram = {"1": completed_events, "2": 0, "3": 0, "4": 0}
    histogram = _require_exact_keys(
        fixed32["batch_histogram"],
        frozenset(expected_histogram),
        "final fixed32 batch histogram",
    )
    for batch, count in histogram.items():
        _require_nonnegative_int(count, f"final fixed32 batch histogram.{batch}")

    _require_nonnegative_float(sfwd["gpu_seconds"], "final SFWD gpu_seconds")
    _require_nonnegative_float(sfwd["wall_seconds"], "final SFWD wall_seconds")
    for key in ("steps", "drafts", "wall_drafts", "wall_steps", "wall_rejected"):
        _require_nonnegative_int(sfwd[key], f"final SFWD {key}")
    for label, span in (("DFWD", dfwd), ("CFWD", cfwd)):
        _require_nonnegative_float(span["gpu_seconds"], f"final {label} gpu_seconds")
        _require_nonnegative_int(span["spans"], f"final {label} spans")

    if (
        final_flush["schema"] != "fr13-fixed32-flush-client-result-v1"
        or ack["schema"] != "fr13-fixed32-flush-ack-v1"
        or ack["mode"] != EXPECTED_MODE
        or ack["status"] != "ok"
        or ack["action"] != "final"
        or ack["generation"] != live_payload["flush_generation"]
        or ack["nonce"] != live_payload["flush_nonce"]
        or ack["producer_pid"] != live_payload["producer_pid"]
        or boundary["schema"] != "fr13-fixed32-boundary-snapshot-v4"
        or boundary["mode"] != EXPECTED_MODE
        or boundary["action"] != "final"
        or boundary["generation"] != live_payload["flush_generation"]
        or boundary["nonce"] != live_payload["flush_nonce"]
        or boundary["producer_pid"] != live_payload["producer_pid"]
        or boundary_counters != ack_counters
        or fixed32["pure_decode_forward_steps"] != completed_events
        or fixed32["complete_work_census_events"] != completed_events
        or fixed32["spec_drafts"] != completed_events
        or fixed32["complete_spec_rows"] != completed_events
        or fixed32["spec_tokens"] != completed_events * 31
        or histogram != expected_histogram
        or fixed32["first_forward_step"] != 0
        or fixed32["last_forward_step"] != completed_events - 1
        or fixed32["events_sha256"] != live_payload["events_sha256"]
        or sfwd["steps"] != completed_events
        or sfwd["drafts"] != completed_events
        or sfwd["wall_steps"] > completed_events
        or dfwd["spans"] != completed_events
        or cfwd["spans"] != completed_events
        or _digest_bytes(boundary_raw) != live_payload["boundary_snapshot_sha256"]
    ):
        raise ValueError("draft-head M32 terminal flush evidence drifted")
    return {
        "boundary_snapshot_sha256": _digest_bytes(boundary_raw),
        "completed_events": completed_events,
        "events_sha256": live_payload["events_sha256"],
        "flush_generation": live_payload["flush_generation"],
    }


def validate_chat_traffic_audit(
    *, audit_path: Path, expected_events: int
) -> dict[str, str | int]:
    expected_events = _require_positive_int(
        expected_events, "expected chat-traffic events"
    )
    audit, raw = load_json(audit_path)
    _require_exact_keys(audit, TRAFFIC_AUDIT_KEYS, "chat traffic audit")
    subset = _require_exact_keys(
        audit["subset"], frozenset({"sha256", "task_count", "task_ids"}), "audit subset"
    )
    checks = _require_exact_keys(audit["checks"], TRAFFIC_CHECK_KEYS, "audit checks")
    stream = _require_exact_keys(
        audit["complete_stream"],
        frozenset(
            {
                "pure_decode_forward_steps",
                "complete_work_census_events",
                "merged_forward_step_intervals",
            }
        ),
        "audit complete stream",
    )
    if not all(value is True for value in checks.values()):
        raise ValueError("chat traffic audit checks are not all true")
    _require_sha256(subset["sha256"], "audit subset sha256")
    _require_positive_int(subset["task_count"], "audit subset task count")
    if (
        not isinstance(subset["task_ids"], list)
        or any(not isinstance(value, str) for value in subset["task_ids"])
    ):
        raise ValueError("audit subset task IDs are malformed")
    for key in ("pure_decode_forward_steps", "complete_work_census_events"):
        _require_positive_int(stream[key], f"audit complete stream {key}")
    intervals = stream["merged_forward_step_intervals"]
    if (
        not isinstance(intervals, list)
        or any(
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(value) is not int or value < 0 for value in interval)
            for interval in intervals
        )
    ):
        raise ValueError("audit complete stream intervals are malformed")

    fetch = _validate_artifact_identity(audit["offload_fetch_status"], "audit fetch")
    proxy_runtime = _require_exact_keys(
        audit["proxy_runtime"],
        frozenset(
            {
                "path",
                "sha256",
                "bytes",
                "canonical_task_set_sha256",
                "raw_dump_environment_absent",
                "raw_dump_artifacts_absent",
            }
        ),
        "audit proxy runtime",
    )
    if not isinstance(proxy_runtime["path"], str) or not proxy_runtime["path"]:
        raise ValueError("audit proxy runtime path is empty")
    _require_sha256(proxy_runtime["sha256"], "audit proxy runtime sha256")
    _require_sha256(
        proxy_runtime["canonical_task_set_sha256"],
        "audit proxy runtime canonical task set",
    )
    _require_positive_int(proxy_runtime["bytes"], "audit proxy runtime bytes")

    ingress = _require_exact_keys(audit["ingress"], INGRESS_KEYS, "audit ingress")
    census = _require_exact_keys(ingress["census"], CENSUS_KEYS, "audit census")
    for label in ("engine", "preflight", "proxy"):
        if not isinstance(ingress[label], dict) or not ingress[label]:
            raise ValueError(f"audit ingress {label} evidence is empty")
    _require_sha256(ingress["canonical_task_set_sha256"], "audit task set")
    _require_sha256(census["sha256"], "audit census sha256")
    for key in ("bytes", "event_count", "request_step_memberships", "successful_engine_requests"):
        _require_positive_int(census[key], f"audit census {key}")
    if not isinstance(census["path"], str) or not census["path"]:
        raise ValueError("audit census path is empty")

    tasks = _require_exact_keys(
        audit["tasks"], frozenset({EXPECTED_INSTANCE}), "audit tasks"
    )
    task = _require_exact_keys(
        tasks[EXPECTED_INSTANCE],
        frozenset(
            {"task_key_id", "dataset_record_sha256", "trace", "task_auth", "terminal", "boundary"}
        ),
        "audit task",
    )
    _require_sha256(task["task_key_id"], "audit task key")
    _require_sha256(task["dataset_record_sha256"], "audit dataset record")
    trace = _require_exact_keys(
        task["trace"],
        frozenset(
            {
                "path",
                "sha256",
                "bytes",
                "event_count",
                "completed_logical_model_requests",
                "model_request_id_sha256s",
                "model_request_ids_sha256",
            }
        ),
        "audit trace",
    )
    _require_sha256(trace["sha256"], "audit trace sha256")
    _require_sha256(trace["model_request_ids_sha256"], "audit trace request digest")
    trace_requests = _require_positive_int(
        trace["completed_logical_model_requests"], "audit trace completed requests"
    )
    _require_positive_int(trace["bytes"], "audit trace bytes")
    _require_positive_int(trace["event_count"], "audit trace events")
    request_digests = trace["model_request_id_sha256s"]
    if (
        not isinstance(request_digests, list)
        or len(request_digests) != trace_requests
        or any(_require_sha256(value, "audit trace request") != value for value in request_digests)
        or request_digests != sorted(request_digests)
    ):
        raise ValueError("audit trace request digest list drifted")

    task_auth = _require_exact_keys(
        task["task_auth"],
        frozenset(
            {
                "completed_logical_model_requests",
                "aborted_logical_requests",
                "accepted_attempts",
                "completed_attempts",
                "failed_attempts",
                "evidence_before_sha256",
                "evidence_after_sha256",
                "evidence_after_ledger_records",
                "evidence_after_ledger_chain_head_sha256",
            }
        ),
        "audit task authentication",
    )
    for key in (
        "completed_logical_model_requests",
        "aborted_logical_requests",
        "accepted_attempts",
        "completed_attempts",
        "failed_attempts",
        "evidence_after_ledger_records",
    ):
        _require_nonnegative_int(task_auth[key], f"audit task authentication {key}")
    for key in (
        "evidence_before_sha256",
        "evidence_after_sha256",
        "evidence_after_ledger_chain_head_sha256",
    ):
        _require_sha256(task_auth[key], f"audit task authentication {key}")

    terminal = _require_exact_keys(
        task["terminal"], frozenset({"agent", "eval", "eval_artifact"}), "audit terminal"
    )
    agent = _require_exact_keys(
        terminal["agent"],
        frozenset({"exit_code", "timed_out", "offloaded", "network_drop"}),
        "audit agent terminal",
    )
    evaluation = _require_exact_keys(
        terminal["eval"],
        frozenset({"verdict", "passed", "harness_exit_code"}),
        "audit eval terminal",
    )
    _validate_artifact_identity(terminal["eval_artifact"], "audit eval artifact")
    boundary = _require_exact_keys(
        task["boundary"],
        frozenset({"path", "sha256", "bytes", "forward_step_interval"}),
        "audit task boundary",
    )
    _require_sha256(boundary["sha256"], "audit task boundary sha256")
    _require_positive_int(boundary["bytes"], "audit task boundary bytes")
    if (
        not isinstance(boundary["path"], str)
        or not boundary["path"]
        or not isinstance(boundary["forward_step_interval"], list)
        or len(boundary["forward_step_interval"]) != 2
        or any(
            type(value) is not int or value < 0
            for value in boundary["forward_step_interval"]
        )
    ):
        raise ValueError("audit task boundary identity is malformed")

    expected_interval = [0, expected_events]
    if (
        audit["schema"] != "fr13-fixed32-chat-task-provenance-audit-v2"
        or audit["mode"] != EXPECTED_MODE
        or audit["dataset_name"] != EXPECTED_DATASET
        or subset
        != {
            "sha256": EXPECTED_B1_SUBSET_SHA256,
            "task_count": 1,
            "task_ids": [EXPECTED_INSTANCE],
        }
        or stream["pure_decode_forward_steps"] != expected_events
        or stream["complete_work_census_events"] != expected_events
        or stream["merged_forward_step_intervals"] != [expected_interval]
        or proxy_runtime["raw_dump_environment_absent"] is not True
        or proxy_runtime["raw_dump_artifacts_absent"] is not True
        or proxy_runtime["canonical_task_set_sha256"]
        != ingress["canonical_task_set_sha256"]
        or ingress["exact_proxy_engine_attempt_parity"] is not True
        or ingress["zero_campaign_rejections"] is not True
        or ingress["zero_failed_or_aborted_requests"] is not True
        or census["all_census_requests_authenticated"] is not True
        or census["all_census_requests_inside_task_brackets"] is not True
        or census["all_successful_requests_present"] is not True
        or census["event_count"] != expected_events
        or census["request_step_memberships"] != expected_events
        or census["per_task_request_step_memberships"]
        != {EXPECTED_INSTANCE: expected_events}
        or census["event_schema"] != "fr13-fixed32-work-census-v11"
        or census["terminal_schema"] != "fr13-fixed32-work-census-terminal-v11"
        or task_auth["completed_logical_model_requests"] != trace_requests
        or task["dataset_record_sha256"] != EXPECTED_DATASET_RECORD_SHA256
        or task_auth["aborted_logical_requests"] != 0
        or task_auth["failed_attempts"] != 0
        or task_auth["accepted_attempts"] != task_auth["completed_attempts"]
        or task_auth["completed_attempts"] < trace_requests
        or agent
        != {"exit_code": 0, "timed_out": False, "offloaded": True, "network_drop": False}
        or evaluation["verdict"] not in {"resolved", "failed"}
        or type(evaluation["passed"]) is not bool
        or evaluation["passed"] is not (evaluation["verdict"] == "resolved")
        or type(evaluation["harness_exit_code"]) is not int
        or boundary["forward_step_interval"] != expected_interval
    ):
        raise ValueError("chat traffic audit provenance drifted")
    return {
        "chat_traffic_audit_sha256": _digest_bytes(raw),
        "completed_events": expected_events,
        "trace_completed_logical_model_requests": trace_requests,
        "fetch_status_sha256": fetch["sha256"],
    }


def validate_rebuilt_chat_traffic_audit(
    *, audit_path: Path, repo: Path
) -> None:
    repo = repo.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    try:
        from fr13_floor_gate import (
            build_fixed32_chat_traffic_audit,
            validate_fixed32_run_subset,
        )
    except ImportError as error:
        raise ValueError(
            "canonical fixed32 traffic-audit validator is unavailable"
        ) from error
    subset = validate_fixed32_run_subset(
        repo / "config/fr13_fixed32/subset_b1_diagnostic_one.json",
        b1_diagnostic=True,
    )
    expected = build_fixed32_chat_traffic_audit(
        audit_path.parent,
        mode=EXPECTED_MODE,
        subset=subset,
        dataset_record_digests={
            EXPECTED_INSTANCE: EXPECTED_DATASET_RECORD_SHA256
        },
    )
    actual, _ = load_json(audit_path)
    if actual != expected:
        raise ValueError("chat traffic audit differs from canonical exact rebuild")


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    candidate_source: Path,
    expected_candidate_source_sha256: str,
    out: Path,
    repo: Path | None = None,
) -> dict[str, Any]:
    expected_live_sha256 = _require_sha256(expected_live_sha256, "live result")
    expected_candidate_source_sha256 = _require_sha256(
        expected_candidate_source_sha256, "qualified candidate source"
    )
    require_regular_file(candidate_source, "candidate source")
    if sha256_file(candidate_source) != expected_candidate_source_sha256:
        raise ValueError("candidate source SHA-256 mismatch")
    if repo is not None:
        validate_rebuilt_chat_traffic_audit(
            audit_path=chat_traffic_audit,
            repo=repo,
        )
    live_payload, live_raw = load_json(live_result)
    if _digest_bytes(live_raw) != expected_live_sha256:
        raise ValueError("live result raw SHA-256 mismatch")
    summary = validate_live_result(
        live_payload,
        expected_source_sha256=expected_candidate_source_sha256,
    )
    terminal = validate_live_evidence(
        live_payload=live_payload,
        final_flush_path=final_flush,
        boundary_snapshot_path=boundary_snapshot,
    )
    traffic = validate_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        expected_events=int(summary["completed_events"]),
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "live_gate_schema": LIVE_SCHEMA,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest_bytes(
            canonical_bytes(live_payload)
        ),
        "instance_id": EXPECTED_INSTANCE,
        "qualified_source_commit": summary["source_commit"],
        "qualified_candidate_source_sha256": expected_candidate_source_sha256,
        "qualified_completed_events": summary["completed_events"],
        "qualified_events_sha256": terminal["events_sha256"],
        "qualified_flush_generation": terminal["flush_generation"],
        "final_flush_sha256": sha256_file(final_flush),
        "boundary_snapshot_sha256": terminal["boundary_snapshot_sha256"],
        "chat_traffic_audit_sha256": traffic["chat_traffic_audit_sha256"],
        "qualified_trace_completed_logical_model_requests": traffic[
            "trace_completed_logical_model_requests"
        ],
        "candidate": EXPECTED_CANDIDATE,
        "geometry": EXPECTED_GEOMETRY,
        "required_runtime": "fixed32 B1 full drafter graph",
        "production_scope": "five exact root64 BF16 draft heads per event",
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest_bytes(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace draft-head pass sidecar: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_bytes(sidecar) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return sidecar


def verify_sidecar(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    expected_live_sha256: str,
    candidate_source: Path,
    expected_candidate_source_sha256: str,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "draft-head pass sidecar"
    )
    expected_candidate_source_sha256 = _require_sha256(
        expected_candidate_source_sha256, "qualified candidate source"
    )
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "qualified live result"
    )
    require_regular_file(candidate_source, "candidate source")
    payload, raw = load_json(sidecar_path)
    if _digest_bytes(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    required = {
        "schema",
        "status",
        "live_gate_schema",
        "live_result_sha256",
        "live_result_canonical_sha256",
        "instance_id",
        "qualified_source_commit",
        "qualified_candidate_source_sha256",
        "qualified_completed_events",
        "qualified_events_sha256",
        "qualified_flush_generation",
        "final_flush_sha256",
        "boundary_snapshot_sha256",
        "chat_traffic_audit_sha256",
        "qualified_trace_completed_logical_model_requests",
        "candidate",
        "geometry",
        "required_runtime",
        "production_scope",
        "canonical_sha256",
    }
    if set(payload) != required:
        raise ValueError("pass sidecar key set drifted")
    canonical_sha256 = payload.pop("canonical_sha256")
    if _require_sha256(canonical_sha256, "sidecar canonical") != _digest_bytes(
        canonical_bytes(payload)
    ):
        raise ValueError("pass sidecar canonical digest mismatch")
    if (
        payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("qualified_candidate_source_sha256")
        != expected_candidate_source_sha256
        or payload.get("live_result_sha256") != expected_live_sha256
        or type(payload.get("qualified_completed_events")) is not int
        or payload["qualified_completed_events"] < 1
        or type(payload.get("qualified_flush_generation")) is not int
        or payload["qualified_flush_generation"] < 1
        or type(payload.get("qualified_trace_completed_logical_model_requests"))
        is not int
        or payload["qualified_trace_completed_logical_model_requests"] < 1
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("required_runtime") != "fixed32 B1 full drafter graph"
        or payload.get("production_scope")
        != "five exact root64 BF16 draft heads per event"
    ):
        raise ValueError("pass sidecar contract drifted")
    _require_commit(payload.get("qualified_source_commit"), "qualified source")
    _require_sha256(payload.get("live_result_sha256"), "live result")
    _require_sha256(
        payload.get("live_result_canonical_sha256"), "live result canonical"
    )
    _require_sha256(payload.get("qualified_events_sha256"), "qualified events")
    _require_sha256(payload.get("final_flush_sha256"), "final flush")
    _require_sha256(
        payload.get("boundary_snapshot_sha256"), "boundary snapshot"
    )
    _require_sha256(
        payload.get("chat_traffic_audit_sha256"), "chat traffic audit"
    )
    if sha256_file(candidate_source) != expected_candidate_source_sha256:
        raise ValueError("attested candidate source SHA-256 mismatch")
    payload["canonical_sha256"] = canonical_sha256
    return payload


def validate_engagement(
    *,
    engagement_path: Path,
    expected_source_sha256: str,
    expected_sidecar_sha256: str,
) -> dict[str, Any]:
    expected_source_sha256 = _require_sha256(
        expected_source_sha256, "engaged candidate source"
    )
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "engaged production sidecar"
    )
    payload, _ = load_json(engagement_path)
    if (
        set(payload) != ENGAGEMENT_KEYS
        or payload.get("schema") != ENGAGEMENT_SCHEMA
        or payload.get("status") != "ENGAGED"
        or payload.get("candidate_source_sha256") != expected_source_sha256
        or payload.get("production_pass_sidecar_sha256")
        != expected_sidecar_sha256
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("selected_root_calls") != 1
        or payload.get("captured_loop_calls") != 4
        or payload.get("fallback_calls") != 0
        or type(payload.get("drafter_graph_id")) is not int
        or payload["drafter_graph_id"] < 1
        or payload.get("drafter_graph_signature")
        != EXPECTED_GRAPH_SIGNATURE
        or payload.get("observed_measured_replays_at_least") != 1
        or payload.get("capture_origin") not in {"measured", "unmeasured"}
        or payload.get("execution_basis") != "cudagraph_replay"
        or type(payload.get("forward_step_index")) is not int
        or payload["forward_step_index"] < 0
        or payload.get("runtime_mode") != "FULL"
    ):
        raise ValueError("draft-head M32 production engagement drifted")
    _require_commit(payload.get("source_commit"), "engagement source")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_live = subparsers.add_parser("validate-live")
    validate_live.add_argument("--live-result", required=True, type=Path)
    validate_live.add_argument("--expected-live-sha256", required=True)
    validate_live.add_argument("--final-flush", required=True, type=Path)
    validate_live.add_argument("--boundary-snapshot", required=True, type=Path)
    validate_live.add_argument("--chat-traffic-audit", required=True, type=Path)
    validate_live.add_argument("--candidate-source", required=True, type=Path)
    validate_live.add_argument("--expected-candidate-source-sha256", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--live-result", required=True, type=Path)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--final-flush", required=True, type=Path)
    issue.add_argument("--boundary-snapshot", required=True, type=Path)
    issue.add_argument("--chat-traffic-audit", required=True, type=Path)
    issue.add_argument("--candidate-source", required=True, type=Path)
    issue.add_argument("--expected-candidate-source-sha256", required=True)
    issue.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--sidecar", required=True, type=Path)
    verify.add_argument("--expected-sidecar-sha256", required=True)
    verify.add_argument("--expected-live-sha256", required=True)
    verify.add_argument("--candidate-source", required=True, type=Path)
    verify.add_argument("--expected-candidate-source-sha256", required=True)
    engagement = subparsers.add_parser("engagement")
    engagement.add_argument("--engagement", required=True, type=Path)
    engagement.add_argument("--expected-source-sha256", required=True)
    engagement.add_argument("--expected-sidecar-sha256", required=True)
    args = parser.parse_args()

    if args.command == "validate-live":
        payload, raw = load_json(args.live_result)
        if _digest_bytes(raw) != _require_sha256(
            args.expected_live_sha256, "live result"
        ):
            raise ValueError("live result raw SHA-256 mismatch")
        require_regular_file(args.candidate_source, "candidate source")
        if sha256_file(args.candidate_source) != args.expected_candidate_source_sha256:
            raise ValueError("candidate source SHA-256 mismatch")
        result = validate_live_result(
            payload,
            expected_source_sha256=args.expected_candidate_source_sha256,
        )
        result.update(
            validate_live_evidence(
                live_payload=payload,
                final_flush_path=args.final_flush,
                boundary_snapshot_path=args.boundary_snapshot,
            )
        )
        result.update(
            validate_chat_traffic_audit(
                audit_path=args.chat_traffic_audit,
                expected_events=int(result["completed_events"]),
            )
        )
        validate_rebuilt_chat_traffic_audit(
            audit_path=args.chat_traffic_audit,
            repo=args.candidate_source.resolve(strict=True).parents[1],
        )
    elif args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            final_flush=args.final_flush,
            boundary_snapshot=args.boundary_snapshot,
            chat_traffic_audit=args.chat_traffic_audit,
            candidate_source=args.candidate_source,
            expected_candidate_source_sha256=args.expected_candidate_source_sha256,
            out=args.out,
            repo=args.candidate_source.resolve(strict=True).parents[1],
        )
    elif args.command == "verify":
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            expected_live_sha256=args.expected_live_sha256,
            candidate_source=args.candidate_source,
            expected_candidate_source_sha256=args.expected_candidate_source_sha256,
        )
    else:
        result = validate_engagement(
            engagement_path=args.engagement,
            expected_source_sha256=args.expected_source_sha256,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
        )
    print(canonical_bytes(result).decode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
