#!/usr/bin/env python3
"""Validate the one-real-SWE fixed32 K64/root1 FP8 drafter-head gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any

try:
    from .fr13_draft_head_m32_pass import (
        validate_chat_traffic_audit,
        validate_live_evidence,
        validate_rebuilt_chat_traffic_audit,
    )
    from . import fr13_qrow16_pass_sidecar as qrow16
except ImportError:
    from fr13_draft_head_m32_pass import (
        validate_chat_traffic_audit,
        validate_live_evidence,
        validate_rebuilt_chat_traffic_audit,
    )
    import fr13_qrow16_pass_sidecar as qrow16


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
_CANDIDATE_COMMON = {
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


def candidate_contract(static_io: bool) -> dict[str, Any]:
    return {
        **_CANDIDATE_COMMON,
        "activation_io": (
            "static_preallocated_raw_out_ops"
            if static_io
            else "wrapper_allocated"
        ),
    }


CANDIDATE = candidate_contract(False)
STATIC_IO_CANDIDATE = candidate_contract(True)
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
QROW16_SO_SHA256 = (
    "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
)
QROW16_SO_BYTES = 299_507_792
QROW16_LIVE_PASS_SHA256 = (
    "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77"
)
ENGAGEMENT_KEYS = {
    "schema",
    "status",
    "arm",
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


def _require_arm(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or any(
            char
            not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for char in value
        )
    ):
        raise ValueError(f"{label} is not a canonical arm name")
    return value


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _validate_engagement(
    payload: dict[str, Any],
    *,
    source_sha: str,
    source_commit: str,
    expected_arm: str,
    expected_static_io: bool = False,
) -> None:
    if set(payload) != ENGAGEMENT_KEYS:
        raise ValueError("FP8 engagement key set drifted")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "ENGAGED"
        or payload.get("arm") != expected_arm
        or payload.get("source_commit") != source_commit
        or payload.get("candidate_source_sha256") != source_sha
        or payload.get("served_batch_size") != 1
        or payload.get("geometry") != GEOMETRY
        or payload.get("candidate")
        != candidate_contract(expected_static_io)
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


def _validate_acceptance(
    payload: dict[str, Any],
    *,
    expected_static_io: bool = False,
) -> dict[str, float | int | str]:
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
        or payload.get("draft_vocab_k") != 65_536
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_head_fp8") is not True
        or payload.get("draft_head_fp8_static_io") is not expected_static_io
        or payload.get("floor_is_full_step_hardware_floor") is not False
        or payload.get("floor_reference_scope")
        != "fixed32_mandatory_weight_read_or_row_compute_lower_bound"
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
    arm = _require_arm(payload.get("arm"), "FP8 acceptance arm")
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
        "arm": arm,
        "events": int(events),
        "accepted_drafts": int(accepted),
        "accepted_drafts_per_event": accept_per_event,
        "committed_tokens_per_event": committed,
    }


def validate_qrow16_production(
    *,
    sidecar_path: Path,
    capture_path: Path,
    candidate_so: Path,
    label: str,
) -> dict[str, Any]:
    _regular(candidate_so, f"{label} Qrow16 candidate SO")
    if (
        candidate_so.stat().st_size != QROW16_SO_BYTES
        or _file_sha256(candidate_so) != QROW16_SO_SHA256
    ):
        raise ValueError(f"{label} Qrow16 candidate identity drifted")
    _regular(sidecar_path, f"{label} Qrow16 production sidecar")
    sidecar_sha = _file_sha256(sidecar_path)
    try:
        sidecar = qrow16.verify_sidecar(
            sidecar_path=sidecar_path,
            expected_sidecar_sha256=sidecar_sha,
            candidate_so=candidate_so,
            expected_candidate_sha256=QROW16_SO_SHA256,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{label} Qrow16 production sidecar is invalid: {error}"
        ) from error
    if sidecar.get("live_result_sha256") != QROW16_LIVE_PASS_SHA256:
        raise ValueError(f"{label} Qrow16 sidecar is not bound to the live PASS")

    capture, capture_raw = _load(capture_path, f"{label} Qrow16 capture")
    required = {
        "schema": "fr13.fixed32.fa2_qrow16_production_capture.v1",
        "status": "ENGAGED",
        "runtime_mode": "FULL",
        "batch_size": 1,
        "layer_count": 16,
        "candidate_so_sha256": QROW16_SO_SHA256,
        "pass_sidecar_sha256": sidecar_sha,
        "dispatch": "qrow16 exact geometry; no fallback",
    }
    if set(capture) != {*required, "graph_id", "graph_signature", "layers"}:
        raise ValueError(f"{label} Qrow16 capture key set drifted")
    for key, expected in required.items():
        if capture.get(key) != expected:
            raise ValueError(f"{label} Qrow16 capture {key} drifted")
    graph_id = capture.get("graph_id")
    graph_signature = capture.get("graph_signature")
    layers = capture.get("layers")
    if (
        isinstance(graph_id, bool)
        or not isinstance(graph_id, int)
        or graph_id <= 0
        or not isinstance(graph_signature, str)
        or re.fullmatch(r"[0-9a-f]{64}", graph_signature) is None
        or not isinstance(layers, list)
        or len(layers) != 16
        or len(set(layers)) != 16
        or any(not isinstance(layer, str) or not layer for layer in layers)
    ):
        raise ValueError(f"{label} Qrow16 graph identity drifted")
    return {
        "candidate_so_sha256": QROW16_SO_SHA256,
        "candidate_so_bytes": QROW16_SO_BYTES,
        "live_pass_sha256": QROW16_LIVE_PASS_SHA256,
        "production_sidecar_sha256": sidecar_sha,
        "production_capture_sha256": _sha256(capture_raw),
        "graph_signature": graph_signature,
        "layer_count": 16,
        "dispatch": required["dispatch"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engagement", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    parser.add_argument("--qrow16-sidecar", type=Path, required=True)
    parser.add_argument("--qrow16-capture", type=Path, required=True)
    parser.add_argument("--qrow16-so", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--expected-static-io", choices=("0", "1"), default="0"
    )
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
    acceptance_summary = _validate_acceptance(
        acceptance,
        expected_static_io=args.expected_static_io == "1",
    )
    _validate_engagement(
        engagement,
        source_sha=source_sha,
        source_commit=source_commit,
        expected_arm=str(acceptance_summary["arm"]),
        expected_static_io=args.expected_static_io == "1",
    )

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
    qrow16_production = validate_qrow16_production(
        sidecar_path=args.qrow16_sidecar,
        capture_path=args.qrow16_capture,
        candidate_so=args.qrow16_so,
        label="gate",
    )

    result = {
        "schema": "fr13.fixed32.draft_head_fp8_real_b1_gate.v2",
        "status": "PASS",
        "classification": "one_real_swe_verified_b1_integrity_gate",
        "performance_tuning_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "suite": "SWE-Verified",
        "instance_id": INSTANCE,
        "arm": acceptance_summary["arm"],
        "source_commit": source_commit,
        "candidate_source_sha256": source_sha,
        "static_io": args.expected_static_io == "1",
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
            "activation_io": engagement["candidate"]["activation_io"],
        },
        "acceptance_telemetry": acceptance_summary,
        "terminal": terminal,
        "traffic": traffic,
        "qrow16_production": qrow16_production,
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
            "schema": "fr13.fixed32.draft_head_fp8_promotion_credential.v2",
            "status": "PASS",
            "qualification_scope": "exact4_real_swe_verified_timing_only",
            "performance_tuning_eligible": True,
            "formal_floor_acceptance_eligible": False,
            "source_commit": source_commit,
            "candidate_source_sha256": source_sha,
            "static_io": gate_result["static_io"],
            "qualification_arm": gate_result["arm"],
            "gate_result_sha256": expected_gate_sha,
            "gate_evidence_sha256": {
                "engagement": _sha256(engagement_raw),
                "acceptance_telemetry": _sha256(acceptance_raw),
                "final_flush": _sha256(final_flush_raw),
                "boundary_snapshot": _sha256(boundary_raw),
                "chat_traffic_audit": _file_sha256(
                    args.chat_traffic_audit
                ),
                "qrow16_sidecar": _file_sha256(args.qrow16_sidecar),
                "qrow16_capture": _file_sha256(args.qrow16_capture),
            },
            "engagement": gate_result["engagement"],
            "qrow16_production": gate_result["qrow16_production"],
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
