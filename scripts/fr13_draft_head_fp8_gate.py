#!/usr/bin/env python3
"""Validate the one-real-SWE fixed32 K64/root1 FP8 drafter-head gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
from pathlib import Path
from typing import Any

try:
    from .fr13_draft_head_m32_pass import (
        validate_chat_traffic_audit,
        validate_live_evidence,
        validate_rebuilt_chat_traffic_audit,
    )
except ImportError:
    from fr13_draft_head_m32_pass import (
        validate_chat_traffic_audit,
        validate_live_evidence,
        validate_rebuilt_chat_traffic_audit,
    )


SCHEMA = "fr13.fixed32.draft_head_fp8_engagement.v1"
INSTANCE = "astropy__astropy-12907"
GRAPH_SIGNATURE = (
    "d9a4ddece41d146e9949b9f8ff7c2603"
    "b8948d157b28ef69244e44469b36150c"
)
GEOMETRY = {
    "calls_per_event": 5,
    "input_hidden": 5120,
    "vocab_rows": 65536,
    "weight_shape": [65536, 5120],
    "weight_stride": [5120, 1],
    "weight_scale_shape": [512, 40],
    "weight_scale_stride": [40, 1],
    "weight_block": [128, 128],
    "activation_group": 128,
}
CANDIDATE = {
    "operation": "vllm_cutlass_block_fp8_scaled_mm",
    "device": "sm121",
    "weight_dtype_bytes": 1,
    "weight_scale_dtype": "torch.float32",
    "activation_scale_layout": "column_major",
    "use_ue8m0": False,
    "output_dtype": "torch.bfloat16",
    "proposal_logits_source": "fp8_output_direct",
    "bf16_shadow_calls": 0,
}
TRAFFIC = {
    "bf16_weight_bytes_per_call_removed": 671_088_640,
    "fp8_weight_bytes_per_call": 335_544_320,
    "fp32_weight_scale_bytes_per_call": 81_920,
    "mandatory_bytes_per_call": 335_626_240,
    "mandatory_bytes_per_event": 1_678_131_200,
    "retained_candidate_bytes": 335_626_240,
    "baseline_mandatory_bytes_per_event": 32_666_638_208,
    "candidate_mandatory_bytes_per_event": 30_989_326_208,
    "floor_bandwidth_gbps": 273,
    "candidate_weight_floor_ms": 113.514015414,
    "one_sided_u95_cap_ms": 130.541117726,
}
ENGAGEMENT_KEYS = {
    "schema",
    "status",
    "source_commit",
    "candidate_source_sha256",
    "served_batch_size",
    "geometry",
    "candidate",
    "traffic",
    "selected_root_calls",
    "captured_loop_calls",
    "fallback_calls",
    "drafter_graph_id",
    "drafter_graph_signature",
    "observed_measured_replays_at_least",
    "capture_origin",
    "execution_basis",
    "forward_step_index",
    "runtime_mode",
    "steady_state_synchronizations",
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


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase commit")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _validate_engagement(
    payload: dict[str, Any], *, source_sha: str, source_commit: str
) -> None:
    if set(payload) != ENGAGEMENT_KEYS:
        raise ValueError("FP8 engagement key set drifted")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "ENGAGED"
        or payload.get("source_commit") != source_commit
        or payload.get("candidate_source_sha256") != source_sha
        or payload.get("served_batch_size") != 1
        or payload.get("geometry") != GEOMETRY
        or payload.get("candidate") != CANDIDATE
        or payload.get("traffic") != TRAFFIC
        or payload.get("selected_root_calls") != 1
        or payload.get("captured_loop_calls") != 4
        or payload.get("fallback_calls") != 0
        or type(payload.get("drafter_graph_id")) is not int
        or payload["drafter_graph_id"] < 1
        or payload.get("drafter_graph_signature") != GRAPH_SIGNATURE
        or payload.get("observed_measured_replays_at_least") != 1
        or payload.get("capture_origin") not in {"measured", "unmeasured"}
        or payload.get("execution_basis") != "cudagraph_replay"
        or type(payload.get("forward_step_index")) is not int
        or payload["forward_step_index"] < 0
        or payload.get("runtime_mode") != "FULL"
        or payload.get("steady_state_synchronizations") != 0
    ):
        raise ValueError("FP8 engagement contract drifted")


def _validate_acceptance(payload: dict[str, Any]) -> dict[str, float | int]:
    engagement = payload.get("engagement")
    per_task = payload.get("per_task")
    raw = payload.get("raw_counter_delta_aggregate")
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("kind") != "speed"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != 1
        or payload.get("task_instance_ids") != [INSTANCE]
        or not isinstance(engagement, dict)
        or engagement.get("tok_per_draft") != 31.0
        or engagement.get("expected_tok_per_draft") != 31.0
        or engagement.get("engaged") is not True
        or not isinstance(per_task, list)
        or len(per_task) != 1
        or per_task[0].get("instance_id") != INSTANCE
        or not isinstance(raw, dict)
        or payload.get("mandatory_weight_bytes") != 30_989_326_208
        or payload.get("weight_floor_ms") != 113.514015414
    ):
        raise ValueError("FP8 acceptance telemetry provenance drifted")
    events = _positive(
        raw.get("vllm:spec_decode_num_drafts_total"), "draft events"
    )
    drafts = _positive(
        raw.get("vllm:spec_decode_num_draft_tokens_total"), "draft tokens"
    )
    accepted = _positive(
        raw.get("vllm:spec_decode_num_accepted_tokens_total"),
        "accepted draft tokens",
    )
    accept_per_event = _positive(
        payload.get("accept_per_event"), "acceptance per event"
    )
    committed = _positive(
        payload.get("committed_per_event"), "committed tokens per event"
    )
    if (
        drafts != events * 31.0
        or not math.isclose(accepted / events, accept_per_event)
        or not math.isclose(committed, accept_per_event + 1.0)
    ):
        raise ValueError("FP8 acceptance accounting drifted")
    return {
        "events": int(events),
        "accepted_drafts": int(accepted),
        "accepted_drafts_per_event": accept_per_event,
        "committed_tokens_per_event": committed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engagement", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path)
    parser.add_argument("--expected-gate-sha256")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_sha = _require_sha(
        args.expected_source_sha256, "expected candidate source"
    )
    source_commit = _require_commit(
        args.expected_source_commit, "expected source commit"
    )
    _regular(args.candidate_source, "candidate source")
    if _file_sha256(args.candidate_source) != source_sha:
        raise ValueError("candidate source SHA-256 mismatch")

    engagement, engagement_raw = _load(args.engagement, "FP8 engagement")
    acceptance, acceptance_raw = _load(args.acceptance, "acceptance telemetry")
    final_flush, final_flush_raw = _load(args.final_flush, "final flush")
    boundary, boundary_raw = _load(args.boundary_snapshot, "boundary snapshot")
    _validate_engagement(
        engagement, source_sha=source_sha, source_commit=source_commit
    )
    acceptance_summary = _validate_acceptance(acceptance)

    ack = final_flush.get("ack")
    metrics = boundary.get("metrics")
    fixed32 = metrics.get("fixed32") if isinstance(metrics, dict) else None
    if not isinstance(ack, dict) or not isinstance(fixed32, dict):
        raise ValueError("terminal fixed32 evidence is malformed")
    completed_events = fixed32.get("complete_work_census_events")
    if type(completed_events) is not int or completed_events < 1:
        raise ValueError("terminal fixed32 event census is empty")
    live_projection = {
        "completed_events": completed_events,
        "flush_generation": ack.get("generation"),
        "flush_nonce": ack.get("nonce"),
        "producer_pid": ack.get("producer_pid"),
        "events_sha256": fixed32.get("events_sha256"),
        "boundary_snapshot_sha256": _sha256(boundary_raw),
    }
    terminal = validate_live_evidence(
        live_payload=live_projection,
        final_flush_path=args.final_flush,
        boundary_snapshot_path=args.boundary_snapshot,
    )
    if acceptance_summary["events"] != completed_events:
        raise ValueError("acceptance and terminal event censuses differ")
    traffic = validate_chat_traffic_audit(
        audit_path=args.chat_traffic_audit,
        expected_events=completed_events,
    )
    validate_rebuilt_chat_traffic_audit(
        audit_path=args.chat_traffic_audit,
        repo=args.repo.resolve(strict=True),
    )

    result = {
        "schema": "fr13.fixed32.draft_head_fp8_real_b1_gate.v1",
        "status": "PASS",
        "classification": "one_real_swe_verified_b1_integrity_gate",
        "performance_tuning_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "suite": "SWE-Verified",
        "instance_id": INSTANCE,
        "source_commit": source_commit,
        "candidate_source_sha256": source_sha,
        "engagement_sha256": _sha256(engagement_raw),
        "acceptance_telemetry_sha256": _sha256(acceptance_raw),
        "final_flush_sha256": _sha256(final_flush_raw),
        "boundary_snapshot_sha256": _sha256(boundary_raw),
        "chat_traffic_audit_sha256": _file_sha256(args.chat_traffic_audit),
        "engagement": {
            "selected_root_calls": 1,
            "captured_loop_calls": 4,
            "fallback_calls": 0,
            "proposal_logits_source": "fp8_output_direct",
            "bf16_shadow_calls": 0,
            "steady_state_synchronizations": 0,
        },
        "acceptance_telemetry": acceptance_summary,
        "terminal": terminal,
        "traffic": traffic,
        "integrity_basis": {
            "canonical_task_terminal": True,
            "authenticated_request_census": True,
            "verifier_and_committer_route": "unchanged_fixed32_rejection_sampling",
            "draft_probs": None,
            "drafter_quality_may_change": True,
        },
        "mandatory_floor": TRAFFIC,
    }
    if args.gate_result is not None:
        expected_gate_sha = _require_sha(
            args.expected_gate_sha256, "expected gate result"
        )
        gate_result, gate_result_raw = _load(
            args.gate_result, "FP8 real-B1 gate result"
        )
        if _sha256(gate_result_raw) != expected_gate_sha:
            raise ValueError("FP8 real-B1 gate result SHA-256 mismatch")
        if gate_result != result:
            raise ValueError(
                "FP8 real-B1 gate result does not match rebuilt raw evidence"
            )
        result = {
            "schema": "fr13.fixed32.draft_head_fp8_promotion_credential.v1",
            "status": "PASS",
            "qualification_scope": "exact4_real_swe_verified_timing_only",
            "performance_tuning_eligible": True,
            "formal_floor_acceptance_eligible": False,
            "source_commit": source_commit,
            "candidate_source_sha256": source_sha,
            "gate_result_sha256": expected_gate_sha,
            "gate_evidence_sha256": {
                "engagement": _sha256(engagement_raw),
                "acceptance_telemetry": _sha256(acceptance_raw),
                "final_flush": _sha256(final_flush_raw),
                "boundary_snapshot": _sha256(boundary_raw),
                "chat_traffic_audit": _file_sha256(
                    args.chat_traffic_audit
                ),
            },
            "engagement": gate_result["engagement"],
            "mandatory_floor": TRAFFIC,
        }
    elif args.expected_gate_sha256 is not None:
        raise ValueError(
            "--expected-gate-sha256 requires --gate-result"
        )

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding="ascii")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
