#!/usr/bin/env python3
"""Validate the reduced packed x-gather gate and timing engagement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_prior_reuse_packed_xgather_rowgroup32_c64_w16_v1"
CANDIDATE_SOURCE_SHA256 = (
    "42fc6ae355a268cb33b454d02914862b2af7fb6b665d808d8899533992750623"
)
CANDIDATE_KERNEL_SOURCE_SHA256 = (
    "ff36101628cc15ead6fef6a7d17c2eb6decbc910c110c635b96d059fea1c1203"
)
REDUCED_GATE_SHA256 = "46c7556b26356b0d53d83b5d6143816f0c04de46d142d2225ce0c497bc4dcfa4"
QUALIFIED_SOURCE_COMMIT = "7c9fda4bc643176f43404ddd4d633789fc46ef23"
TASK_MARKER_SHA256 = "04fe7f61a0e0bbd48bf28127385c481b85550b291535f3705511494ba24c8463"
SUPPORT_FILES = {
    "identity_and_lifecycle.json": (
        "7e0a81516fd3f7342cdd680c77030098d6c570bfc7d12d15a53f547ea6896d61"
    ),
    "record_summary.json": (
        "9363407357abe17291e3fabe600d508577466cd41e76986a424f4f8f30aa0e01"
    ),
    "traffic_model.json": (
        "f90980b8c929bb8c717ebdcd254923e693c363deb6df577d77ebb3a2420f2744"
    ),
}


class PassError(ValueError):
    pass


def _strict_load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise PassError(f"{label} cannot be read: {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PassError(f"{label} must be one regular non-symlink file")
    if not raw or len(raw) > 131072:
        raise PassError(f"{label} is empty or exceeds 128 KiB")

    def reject_duplicates(pairs):
        payload = {}
        for key, value in pairs:
            if key in payload:
                raise PassError(f"{label} has duplicate key: {key}")
            payload[key] = value
        return payload

    def reject_nonfinite(value: str):
        raise PassError(f"{label} has non-finite value: {value}")

    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PassError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(payload, dict):
        raise PassError(f"{label} must contain an object")
    return payload, raw


def _sha256(path: Path, label: str) -> str:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise PassError(f"{label} cannot be read: {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PassError(f"{label} must be one regular non-symlink file")
    return hashlib.sha256(raw).hexdigest()


def _require(payload: dict[str, Any], required: dict[str, Any], label: str) -> None:
    mismatches = [key for key, value in required.items() if payload.get(key) != value]
    if mismatches:
        raise PassError(f"{label} contract mismatch: " + ",".join(mismatches))


def _validate_support_files(gate: Path) -> dict[str, str]:
    support_digests = {}
    payloads = {}
    for name, expected_digest in SUPPORT_FILES.items():
        path = gate.parent / name
        payload, raw = _strict_load(path, f"reduced gate support {name}")
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_digest:
            raise PassError(f"reduced gate support identity drifted: {name}")
        support_digests[name] = digest
        payloads[name] = payload

    _require(
        payloads["record_summary.json"],
        {
            "schema": "fr13.fixed32.sfwd_prior_reuse.reduced_record_summary.v1",
            "all_records_real_task_authenticated": True,
            "all_records_reference_always_served": True,
            "all_records_status_pass": True,
            "byte_equal_false": 0,
            "candidate_count": 1,
            "compared_bytes": 30749491200,
            "comparison_records": 22080,
            "differing_bytes": 0,
            "layer_count": 48,
            "shape_or_dtype_mismatches": 0,
            "surface_comparisons": 44160,
            "surface_counts": {
                "commit_source_stage": 22080,
                "conv_out": 22080,
            },
            "zero_diff_false": 0,
        },
        "reduced record summary",
    )
    identity = payloads["identity_and_lifecycle.json"]
    _require(
        identity,
        {"schema": "fr13.fixed32.sfwd_prior_reuse.reduced_identity_lifecycle.v1"},
        "reduced identity/lifecycle",
    )
    for manifest_name in ("source_manifest", "runtime_manifest", "external_manifest"):
        manifest = identity.get(manifest_name)
        if (
            not isinstance(manifest, dict)
            or manifest.get("launch_equals_end") is not True
            or manifest.get("launch_sha256") != manifest.get("end_sha256")
            or re.fullmatch(r"[0-9a-f]{64}", str(manifest.get("launch_sha256", "")))
            is None
        ):
            raise PassError(f"reduced identity/lifecycle {manifest_name} mismatch")
    _require(
        identity.get("post_run_host_census", {}),
        {
            "gpu_compute_processes": 0,
            "gpu_memory_used_mib": 0,
            "running_docker_containers": 0,
        },
        "reduced post-run host census",
    )
    _require(
        payloads["traffic_model.json"],
        {
            "schema": "fr13.fixed32.sfwd_xgather.logical_traffic.v2",
            "scope": "logical kernel operand traffic before cache effects",
            "status": "analytical_not_measured",
            "packed_xgather": {
                "global_bytes_per_cta": 13700,
                "global_bytes_per_request_layer": 2192000,
                "global_x_elements_per_channel": 32,
                "logical_shared_bytes_per_cta": 14592,
            },
        },
        "reduced traffic model",
    )
    return support_digests


def validate_gate(
    gate: Path,
    *,
    expected_gate_sha256: str,
    candidate_source: Path,
    candidate_kernel_source: Path,
) -> dict[str, Any]:
    if expected_gate_sha256 != REDUCED_GATE_SHA256:
        raise PassError("reduced gate identity is not the qualified credential")
    payload, raw = _strict_load(gate, "reduced gate")
    if hashlib.sha256(raw).hexdigest() != expected_gate_sha256:
        raise PassError("reduced gate raw SHA-256 mismatch")
    source_sha256 = _sha256(candidate_source, "candidate launcher source")
    kernel_sha256 = _sha256(candidate_kernel_source, "candidate kernel source")
    if source_sha256 != CANDIDATE_SOURCE_SHA256:
        raise PassError("qualified candidate launcher source SHA-256 drift")
    if kernel_sha256 != CANDIDATE_KERNEL_SOURCE_SHA256:
        raise PassError("qualified candidate kernel source SHA-256 drift")
    _require(
        payload,
        {
            "schema": "fr13.fixed32.sfwd_prior_reuse.reduced_b1_byte_pass.v1",
            "status": "pass_source_only",
            "run_classification": "one_real_swe_verified_k64_root_b1_byte_diagnostic",
            "candidate": CANDIDATE,
            "source_commit": QUALIFIED_SOURCE_COMMIT,
            "candidate_source_sha256": CANDIDATE_SOURCE_SHA256,
            "candidate_kernel_source_sha256": CANDIDATE_KERNEL_SOURCE_SHA256,
            "candidate_conv_launches_per_layer": 1,
            "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
            "comparison_records": 22080,
            "contains_raw_logs": False,
            "contains_task_or_model_content": False,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "floor_acceptance_eligible": False,
            "layer_count": 48,
            "no_fallback": True,
            "physical_rows_per_request": 32,
            "production_enabled": False,
            "real_task_authenticated": True,
            "reference_always_served": True,
            "source_descriptor_device_validation": False,
            "source_descriptor_launcher_argument": False,
            "swe_orchestrator_exit_code": 0,
            "task_count": 1,
            "task_failure_mode_counts": {"tests_passed": 1},
            "task_verdict_counts": {"resolved": 1},
            "timing_eligible": False,
            "topology_host_validation": "exact_parent_each_launch",
        },
        "reduced gate",
    )
    support_digests = _validate_support_files(gate)
    return {
        "schema": "fr13.fixed32.sfwd_xgather.timing_binding.v1",
        "status": "pass",
        "candidate": CANDIDATE,
        "batch_size": 1,
        "task_marker_sha256": TASK_MARKER_SHA256,
        "reduced_gate_sha256": expected_gate_sha256,
        "candidate_source_sha256": source_sha256,
        "candidate_kernel_source_sha256": kernel_sha256,
        "support_file_sha256": support_digests,
        "candidate_serving_permitted": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }


def validate_engagement(
    path: Path,
    *,
    gate_sha256: str,
    candidate_source: Path,
    candidate_kernel_source: Path,
) -> dict[str, Any]:
    if gate_sha256 != REDUCED_GATE_SHA256:
        raise PassError("timing engagement gate is not the qualified credential")
    payload, _ = _strict_load(path, "timing engagement")
    source_sha256 = _sha256(candidate_source, "candidate launcher source")
    kernel_sha256 = _sha256(candidate_kernel_source, "candidate kernel source")
    if source_sha256 != CANDIDATE_SOURCE_SHA256:
        raise PassError("timing engagement candidate launcher source drifted")
    if kernel_sha256 != CANDIDATE_KERNEL_SOURCE_SHA256:
        raise PassError("timing engagement candidate kernel source drifted")
    _require(
        payload,
        {
            "schema": "fr13.fixed32.sfwd_xgather.timing_engagement.v1",
            "status": "engaged",
            "run_classification": (
                "one_real_swe_verified_k64_root_b1_packed_xgather_timing_diagnostic"
            ),
            "candidate": CANDIDATE,
            "candidate_kernel": (
                "_fr13_fixed32_sfwd_prior_reuse_packed_xgather_kernel"
            ),
            "batch_size": 1,
            "task_marker_sha256": TASK_MARKER_SHA256,
            "layer_count": 48,
            "reduced_gate_sha256": gate_sha256,
            "candidate_source_sha256": source_sha256,
            "candidate_kernel_source_sha256": kernel_sha256,
            "candidate_served": True,
            "sole_conv_source_producer": True,
            "real_task_bound": True,
            "physical_rows_per_request": 32,
            "source_rows_per_request": 36,
            "conv_rows_per_program": 32,
            "conv_block_c": 64,
            "conv_num_warps": 16,
            "draft_vocab_k": 65536,
            "draft_vocab_root": 1,
            "candidate_conv_launches_per_layer": 1,
            "incumbent_conv_launches_per_layer": 0,
            "source_descriptor_device_validation": False,
            "source_descriptor_launcher_argument": False,
            "topology_host_validation": "exact_parent_each_launch",
            "fallback_permitted": False,
            "timing_eligible": False,
            "floor_acceptance_eligible": False,
            "production_eligible": False,
        },
        "timing engagement",
    )
    layer_digest = payload.get("layer_key_digest")
    launches = payload.get("launches_observed")
    if not isinstance(layer_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", layer_digest
    ) is None:
        raise PassError("timing engagement layer-key digest mismatch")
    if isinstance(launches, bool) or not isinstance(launches, int) or launches != 48:
        raise PassError("timing engagement launch count mismatch")
    if "layer_keys" in payload or "task_marker" in payload:
        raise PassError("timing engagement contains raw identifiers")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("validate")
    gate.add_argument("--gate", type=Path, required=True)
    gate.add_argument("--expected-gate-sha256", required=True)
    gate.add_argument("--candidate-source", type=Path, required=True)
    gate.add_argument("--candidate-kernel-source", type=Path, required=True)
    engagement = commands.add_parser("verify-engagement")
    engagement.add_argument("--engagement", type=Path, required=True)
    engagement.add_argument("--expected-gate-sha256", required=True)
    engagement.add_argument("--candidate-source", type=Path, required=True)
    engagement.add_argument("--candidate-kernel-source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_gate(
            args.gate,
            expected_gate_sha256=args.expected_gate_sha256,
            candidate_source=args.candidate_source,
            candidate_kernel_source=args.candidate_kernel_source,
        )
    else:
        result = validate_engagement(
            args.engagement,
            gate_sha256=args.expected_gate_sha256,
            candidate_source=args.candidate_source,
            candidate_kernel_source=args.candidate_kernel_source,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
