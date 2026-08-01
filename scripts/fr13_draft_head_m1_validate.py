#!/usr/bin/env python3
"""Validate the real-B1 shadow result for the BF16 M1 draft head."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

from fr13_draft_head_m32_pass import (
    EXPECTED_B1_SUBSET_SHA256,
    EXPECTED_DATASET_RECORD_SHA256,
    EXPECTED_INSTANCE,
    load_json,
    sha256_file,
    validate_chat_traffic_audit,
    validate_live_evidence,
)


LIVE_SCHEMA = "fr13.fixed32.draft_head_full_m1_live_ab.v1"
VALIDATION_SCHEMA = "fr13.fixed32.draft_head_full_m1_live_validation.v1"
POSITIONS = ("root", "mtp1", "mtp2", "mtp3", "mtp4")
VOCAB_SIZE = 248_320
HEX = frozenset("0123456789abcdef")
EXPECTED_GEOMETRY = {
    "batch_size": 1,
    "supported_batch_sizes": [1],
    "calls_per_event": 5,
    "head_positions": list(POSITIONS),
    "input_shape": [1, 5120],
    "input_stride": [5120, 1],
    "weight_shape": [248320, 5120],
    "weight_stride": [5120, 1],
    "output_shape": [1, 248320],
    "output_stride": [248320, 1],
    "dtype": "torch.bfloat16",
}
EXPECTED_CANDIDATE = {
    "module": "ParallelLMHead",
    "method": "UnquantizedEmbeddingMethod",
    "operation": (
        "custom stock-order BF16 M1 GEMV shadow after stock reference"
    ),
    "gemv_mnk": [1, 248320, 5120],
    "grid": [31040, 1, 1],
    "block": [16, 8, 1],
    "dynamic_shared_bytes": 544,
    "shared_row_stride_floats": 17,
    "k_partition_lanes": 16,
    "lane_k_iterations": 320,
    "reduction_strides": [8, 4, 2, 1],
    "accumulator": "fp32 positive zero",
    "multiply_accumulate": "__fmaf_rn dependent chain",
    "reduction": "__fadd_rn shared-memory tree",
    "epilogue": "__fmaf_rn(1.0f, reduced_sum, 0.0f)",
    "output_conversion": "__float2bfloat16_rn",
    "candidate_launches_per_head": 1,
    "served_rows": 0,
    "shadow_compared_rows": 1,
}
EXPECTED_BUILD_CONTRACT = {
    "grid": [31040, 1, 1],
    "block": [16, 8, 1],
    "dynamic_shared_bytes": 544,
    "shared_row_stride_floats": 17,
    "gemv_mnk": [1, 248320, 5120],
    "k_partition_lanes": 16,
    "lane_k_iterations": 320,
    "reduction_strides": [8, 4, 2, 1],
    "accumulator": "fp32 positive zero",
    "multiply_accumulate": "__fmaf_rn dependent scalar chain",
    "reduction": "__fadd_rn shared-memory tree",
    "epilogue": "__fmaf_rn(1.0f, reduced_sum, 0.0f)",
    "output": "__float2bfloat16_rn",
}
LIVE_KEYS = frozenset(
    {
        "acceptance_eligible",
        "batch_size",
        "bf16_elements_compared",
        "binary",
        "boundary_snapshot_sha256",
        "build_attestation_sha256",
        "candidate",
        "candidate_source_sha256",
        "comparison_view",
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
        "head_positions",
        "instance_id",
        "patcher_sha256",
        "per_head",
        "performance_measurement",
        "producer_pid",
        "production_default_enabled",
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


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in HEX for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in HEX for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase 40-character commit")
    return value


def _require_regular(path: Path, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")


def validate_build_attestation(
    payload: dict[str, Any],
    *,
    expected_source_sha256: str,
    expected_so_sha256: str,
    expected_so_bytes: int,
) -> dict[str, Any]:
    """Require the pinned toolchain record that binds source to the SO."""
    _require_sha256(expected_source_sha256, "candidate source")
    _require_sha256(expected_so_sha256, "candidate SO")
    expected_keys = {
        "binary",
        "byte_equality_claim",
        "cuda_arch",
        "cuda_release",
        "kernel_contract",
        "performance_measurement",
        "production_default_enabled",
        "schema",
        "source",
        "status",
        "torch_version",
    }
    if set(payload) != expected_keys:
        raise ValueError("draft-head M1 build attestation key set drifted")
    if (
        payload.get("schema") != "fr13.fixed32.bf16_gemvx_m1_build.v1"
        or payload.get("status") != "BUILT_UNQUALIFIED"
        or payload.get("performance_measurement") is not False
        or payload.get("byte_equality_claim") is not False
        or payload.get("production_default_enabled") is not False
        or payload.get("torch_version") != "2.10.0+cu130"
        or payload.get("cuda_release") != "13.0"
        or payload.get("cuda_arch") != "12.1a"
        or payload.get("kernel_contract") != EXPECTED_BUILD_CONTRACT
    ):
        raise ValueError("draft-head M1 build attestation contract drifted")
    if payload.get("source") != {
        "path": "csrc/fr13_bf16_gemvx_m1.cu",
        "sha256": expected_source_sha256,
    }:
        raise ValueError("draft-head M1 build source identity drifted")
    binary = payload.get("binary")
    if (
        not isinstance(binary, dict)
        or set(binary) != {"path", "sha256", "bytes", "mode"}
        or not isinstance(binary.get("path"), str)
        or not binary["path"]
        or binary.get("sha256") != expected_so_sha256
        or binary.get("bytes") != expected_so_bytes
        or binary.get("mode") != "0555"
    ):
        raise ValueError("draft-head M1 build binary identity drifted")
    return payload


def validate_live_result(
    payload: dict[str, Any],
    *,
    expected_source_sha256: str,
    expected_patcher_sha256: str,
    expected_build_attestation_sha256: str,
    expected_so_sha256: str,
    expected_so_bytes: int,
) -> dict[str, Any]:
    """Validate exact five-position raw-BF16 equality and provenance."""
    for value, label in (
        (expected_source_sha256, "candidate source"),
        (expected_patcher_sha256, "patcher"),
        (expected_build_attestation_sha256, "build attestation"),
        (expected_so_sha256, "candidate SO"),
    ):
        _require_sha256(value, label)
    if type(expected_so_bytes) is not int or expected_so_bytes < 1:
        raise ValueError("candidate SO bytes must be positive")
    if frozenset(payload) != LIVE_KEYS:
        raise ValueError("draft-head M1 live result key set drifted")
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("draft-head M1 live result is not a PASS record")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("candidate_source_sha256") != expected_source_sha256
        or payload.get("patcher_sha256") != expected_patcher_sha256
        or payload.get("build_attestation_sha256")
        != expected_build_attestation_sha256
        or payload.get("head_positions") != list(POSITIONS)
        or payload.get("comparison_view")
        != "raw torch.int16 over all BF16 logits"
        or payload.get("served_return")
        != "stock reference BF16 logits computed first and unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("acceptance_eligible") is not False
        or payload.get("production_default_enabled") is not False
        or payload.get("finalized_by_fixed32_flush") is not True
        or payload.get("flush_action") != "final"
    ):
        raise ValueError("draft-head M1 live result provenance drifted")
    _require_commit(payload.get("source_commit"), "live source")
    for key in ("events_sha256", "boundary_snapshot_sha256", "flush_nonce"):
        _require_sha256(payload.get(key), f"live {key}")

    binary = payload.get("binary")
    if not isinstance(binary, dict) or set(binary) != {"path", "sha256", "bytes"}:
        raise ValueError("draft-head M1 binary identity drifted")
    if (
        binary.get("path") != "/tmp/fr13_bf16_gemvx_m1.abi3.so"
        or binary.get("sha256") != expected_so_sha256
        or binary.get("bytes") != expected_so_bytes
    ):
        raise ValueError("draft-head M1 binary identity drifted")

    events = payload.get("completed_events")
    if (
        type(events) is not int
        or events < 1
        or payload.get("complete_work_census_events") != events
        or payload.get("work_census_last_event_index") != events - 1
        or type(payload.get("flush_generation")) is not int
        or payload["flush_generation"] < 1
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
    ):
        raise ValueError("draft-head M1 live event census drifted")
    per_head = payload.get("per_head")
    if not isinstance(per_head, list) or len(per_head) != len(POSITIONS):
        raise ValueError("draft-head M1 per-head census drifted")
    for position, record in zip(POSITIONS, per_head, strict=True):
        expected = {
            "position": position,
            "full_logit_comparisons": events,
            "bf16_elements_compared": events * VOCAB_SIZE,
            "raw_bf16_mismatches": 0,
        }
        if record != expected:
            raise ValueError("draft-head M1 per-head census drifted")
    if (
        payload.get("full_logit_comparisons") != events * len(POSITIONS)
        or payload.get("bf16_elements_compared")
        != events * len(POSITIONS) * VOCAB_SIZE
        or payload.get("raw_bf16_mismatches") != 0
    ):
        raise ValueError("draft-head M1 aggregate comparison census drifted")
    return payload


def validate(
    *,
    live_result: Path,
    expected_live_sha256: str,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    candidate_source: Path,
    expected_candidate_source_sha256: str,
    patcher: Path,
    expected_patcher_sha256: str,
    build_attestation: Path,
    expected_build_attestation_sha256: str,
    candidate_so: Path,
    expected_candidate_so_sha256: str,
) -> dict[str, Any]:
    """Bind equality evidence to immutable sources and authenticated B1."""
    for path, label in (
        (live_result, "live result"),
        (final_flush, "final flush"),
        (boundary_snapshot, "boundary snapshot"),
        (chat_traffic_audit, "chat traffic audit"),
        (candidate_source, "candidate source"),
        (patcher, "patcher"),
        (build_attestation, "build attestation"),
        (candidate_so, "candidate SO"),
    ):
        _require_regular(path, label)
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected live result"
    )
    identities = {
        "candidate_source": (
            candidate_source,
            expected_candidate_source_sha256,
        ),
        "patcher": (patcher, expected_patcher_sha256),
        "build_attestation": (
            build_attestation,
            expected_build_attestation_sha256,
        ),
        "candidate_so": (candidate_so, expected_candidate_so_sha256),
    }
    for label, (path, expected_sha) in identities.items():
        _require_sha256(expected_sha, f"expected {label}")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"{label} SHA-256 drifted")
    if sha256_file(live_result) != expected_live_sha256:
        raise ValueError("live result SHA-256 drifted")
    payload, _ = load_json(live_result)
    build_payload, _ = load_json(build_attestation)
    validate_build_attestation(
        build_payload,
        expected_source_sha256=expected_candidate_source_sha256,
        expected_so_sha256=expected_candidate_so_sha256,
        expected_so_bytes=candidate_so.stat().st_size,
    )
    validate_live_result(
        payload,
        expected_source_sha256=expected_candidate_source_sha256,
        expected_patcher_sha256=expected_patcher_sha256,
        expected_build_attestation_sha256=(
            expected_build_attestation_sha256
        ),
        expected_so_sha256=expected_candidate_so_sha256,
        expected_so_bytes=candidate_so.stat().st_size,
    )
    terminal = validate_live_evidence(
        live_payload=payload,
        final_flush_path=final_flush,
        boundary_snapshot_path=boundary_snapshot,
    )
    traffic = validate_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        expected_events=int(payload["completed_events"]),
    )
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "classification": "real_swe_verified_b1_kernel_byte_diagnostic",
        "diagnostic_only": True,
        "performance_measurement": False,
        "acceptance_eligible": False,
        "suite": "SWE-Verified",
        "instance_id": EXPECTED_INSTANCE,
        "subset_sha256": EXPECTED_B1_SUBSET_SHA256,
        "dataset_record_sha256": EXPECTED_DATASET_RECORD_SHA256,
        "source_commit": payload["source_commit"],
        "candidate_source_sha256": expected_candidate_source_sha256,
        "patcher_sha256": expected_patcher_sha256,
        "build_attestation_sha256": expected_build_attestation_sha256,
        "candidate_so_sha256": expected_candidate_so_sha256,
        "candidate_so_bytes": candidate_so.stat().st_size,
        "live_result_sha256": expected_live_sha256,
        "final_flush_sha256": sha256_file(final_flush),
        "boundary_snapshot_sha256": sha256_file(boundary_snapshot),
        "chat_traffic_audit_sha256": sha256_file(chat_traffic_audit),
        "completed_events": payload["completed_events"],
        "bf16_elements_compared": payload["bf16_elements_compared"],
        "head_positions": list(POSITIONS),
        "terminal_complete_work_census_events": terminal[
            "completed_events"
        ],
        "authenticated_traffic_events": traffic["completed_events"],
        "served_return": payload["served_return"],
    }


def validate_build(
    *,
    build_attestation: Path,
    expected_build_attestation_sha256: str,
    candidate_source: Path,
    expected_candidate_source_sha256: str,
    candidate_so: Path,
    expected_candidate_so_sha256: str,
) -> dict[str, Any]:
    """Validate the immutable CPU-side build record before any GPU launch."""
    for path, label in (
        (build_attestation, "build attestation"),
        (candidate_source, "candidate source"),
        (candidate_so, "candidate SO"),
    ):
        _require_regular(path, label)
    for value, label in (
        (expected_build_attestation_sha256, "build attestation"),
        (expected_candidate_source_sha256, "candidate source"),
        (expected_candidate_so_sha256, "candidate SO"),
    ):
        _require_sha256(value, label)
    if sha256_file(build_attestation) != expected_build_attestation_sha256:
        raise ValueError("build attestation SHA-256 drifted")
    if sha256_file(candidate_source) != expected_candidate_source_sha256:
        raise ValueError("candidate source SHA-256 drifted")
    if sha256_file(candidate_so) != expected_candidate_so_sha256:
        raise ValueError("candidate SO SHA-256 drifted")
    payload, _ = load_json(build_attestation)
    validate_build_attestation(
        payload,
        expected_source_sha256=expected_candidate_source_sha256,
        expected_so_sha256=expected_candidate_so_sha256,
        expected_so_bytes=candidate_so.stat().st_size,
    )
    return {
        "schema": "fr13.fixed32.bf16_gemvx_m1_build_validation.v1",
        "status": "PASS",
        "qualification": "build_identity_only",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "build_attestation_sha256": expected_build_attestation_sha256,
        "candidate_source_sha256": expected_candidate_source_sha256,
        "candidate_so_sha256": expected_candidate_so_sha256,
        "candidate_so_bytes": candidate_so.stat().st_size,
        "torch_version": "2.10.0+cu130",
        "cuda_release": "13.0",
        "cuda_arch": "12.1a",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_build_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--build-attestation", type=Path, required=True)
        command.add_argument(
            "--expected-build-attestation-sha256", required=True
        )
        command.add_argument("--candidate-source", type=Path, required=True)
        command.add_argument(
            "--expected-candidate-source-sha256", required=True
        )
        command.add_argument("--candidate-so", type=Path, required=True)
        command.add_argument(
            "--expected-candidate-so-sha256", required=True
        )

    build = subparsers.add_parser("validate-build")
    add_build_inputs(build)
    live = subparsers.add_parser("validate-live")
    add_build_inputs(live)
    live.add_argument("--live-result", type=Path, required=True)
    live.add_argument("--expected-live-sha256", required=True)
    live.add_argument("--final-flush", type=Path, required=True)
    live.add_argument("--boundary-snapshot", type=Path, required=True)
    live.add_argument("--chat-traffic-audit", type=Path, required=True)
    live.add_argument("--patcher", type=Path, required=True)
    live.add_argument("--expected-patcher-sha256", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-build":
            result = validate_build(
                build_attestation=args.build_attestation,
                expected_build_attestation_sha256=(
                    args.expected_build_attestation_sha256
                ),
                candidate_source=args.candidate_source,
                expected_candidate_source_sha256=(
                    args.expected_candidate_source_sha256
                ),
                candidate_so=args.candidate_so,
                expected_candidate_so_sha256=(
                    args.expected_candidate_so_sha256
                ),
            )
        else:
            result = validate(
                live_result=args.live_result,
                expected_live_sha256=args.expected_live_sha256,
                final_flush=args.final_flush,
                boundary_snapshot=args.boundary_snapshot,
                chat_traffic_audit=args.chat_traffic_audit,
                candidate_source=args.candidate_source,
                expected_candidate_source_sha256=(
                    args.expected_candidate_source_sha256
                ),
                patcher=args.patcher,
                expected_patcher_sha256=args.expected_patcher_sha256,
                build_attestation=args.build_attestation,
                expected_build_attestation_sha256=(
                    args.expected_build_attestation_sha256
                ),
                candidate_so=args.candidate_so,
                expected_candidate_so_sha256=(
                    args.expected_candidate_so_sha256
                ),
            )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
