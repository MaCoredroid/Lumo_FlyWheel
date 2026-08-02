#!/usr/bin/env python3
"""Validate the reduced prior-reuse gate and candidate timing engagement."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_prior_reuse_rowgroup32_c64_v1"
CANDIDATE_SOURCE_SHA256 = (
    "2be68a4f0483fccd4a254ec90366ced00590de0301d23255096eceef9bd3eef6"
)
REDUCED_GATE_SHA256 = "eb8c88520e4bc8fc3168049c0a9c0c0fe60893d9cc5d9f6eeee9fb324744b0ce"
VERIFIED_LIVE_PASS_SHA256 = (
    "43f509fd308b74bc29c0fb48116ed915ed6f6ee1bdc2ff2de097281c2d217236"
)
VERIFIED_GATE_SUMMARY_SHA256 = (
    "75384199d975fd4b8d6f787b5b2781a17e422f9a77f1ab79149ac4758fef23c9"
)
TASK_MARKER = "swe_verified:astropy__astropy-12907"


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
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise PassError(f"{label} must be a regular non-symlink file")
    return hashlib.sha256(raw).hexdigest()


def validate_gate(
    gate: Path, *, expected_gate_sha256: str, candidate_source: Path
) -> dict[str, Any]:
    if expected_gate_sha256 != REDUCED_GATE_SHA256:
        raise PassError("reduced gate identity is not the qualified credential")
    payload, raw = _strict_load(gate, "reduced gate")
    if hashlib.sha256(raw).hexdigest() != expected_gate_sha256:
        raise PassError("reduced gate raw SHA-256 mismatch")
    source_sha256 = _sha256(candidate_source, "candidate source")
    if source_sha256 != CANDIDATE_SOURCE_SHA256:
        raise PassError("qualified candidate source SHA-256 drift")
    required = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.k64_root_b1_gate.reduced.v1",
        "status": "pass",
        "run_classification": "one_real_swe_verified_k64_root_b1_byte_diagnostic",
        "candidate": CANDIDATE,
        "source_commit": "b6572a9ab91f281d7c1f84bfb41c24329e6323da",
        "candidate_source_sha256": CANDIDATE_SOURCE_SHA256,
        "qualified_rowgroup8_kernel_preserved": True,
        "task_count": 1,
        "task_outcome": "resolved",
        "task_metrics_pre_present": True,
        "task_metrics_post_present": True,
        "real_task_authenticated": True,
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "conv_rows_per_program": 32,
        "conv_block_c": 64,
        "candidate_conv_launches_per_layer": 1,
        "layer_count": 48,
        "comparisons": 25056,
        "comparisons_per_layer": 522,
        "compared_byte_surface_instances": 50112,
        "compared_bytes": 34893987840,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "mismatching_comparisons": 0,
        "total_differing_bytes": 0,
        "total_shape_or_dtype_mismatches": 0,
        "reference_returned": True,
        "production_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff",
        "k64_gather": {"shim_engagements": 1, "root_engagements": 1, "fallbacks": 0},
        "source_manifest_launch_end_equal": True,
        "runtime_manifest_launch_end_equal": True,
        "external_manifest_launch_end_equal": True,
        "verified_live_pass_sha256": VERIFIED_LIVE_PASS_SHA256,
        "verified_gate_summary_sha256": VERIFIED_GATE_SUMMARY_SHA256,
        "teardown_clean": True,
        "raw_artifacts_included": False,
    }
    mismatches = [key for key, value in required.items() if payload.get(key) != value]
    if mismatches:
        raise PassError("reduced gate contract mismatch: " + ",".join(mismatches))
    return {
        "schema": "fr13.fixed32.sfwd_prior_reuse.timing_binding.v1",
        "status": "pass",
        "candidate": CANDIDATE,
        "batch_size": 1,
        "task_marker": TASK_MARKER,
        "reduced_gate_sha256": expected_gate_sha256,
        "candidate_source_sha256": source_sha256,
        "candidate_serving_permitted": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }


def validate_engagement(
    path: Path, *, gate_sha256: str, candidate_source: Path
) -> dict[str, Any]:
    if gate_sha256 != REDUCED_GATE_SHA256:
        raise PassError("timing engagement gate is not the qualified credential")
    payload, _ = _strict_load(path, "timing engagement")
    source_sha256 = _sha256(candidate_source, "candidate source")
    if source_sha256 != CANDIDATE_SOURCE_SHA256:
        raise PassError("timing engagement candidate source drifted")
    required = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.timing_engagement.v1",
        "status": "engaged",
        "run_classification": "one_real_swe_verified_k64_root_b1_candidate_timing_diagnostic",
        "candidate": CANDIDATE,
        "batch_size": 1,
        "task_marker": TASK_MARKER,
        "layer_count": 48,
        "reduced_gate_sha256": gate_sha256,
        "candidate_source_sha256": source_sha256,
        "candidate_served": True,
        "sole_conv_source_producer": True,
        "real_task_bound": True,
        "physical_rows_per_request": 32,
        "source_rows_per_request": 36,
        "conv_rows_per_program": 32,
        "conv_block_c": 64,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "candidate_conv_launches_per_layer": 1,
        "incumbent_conv_launches_per_layer": 0,
        "fallback_permitted": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    mismatches = [key for key, value in required.items() if payload.get(key) != value]
    layers = payload.get("layer_keys")
    if (
        not isinstance(layers, list)
        or len(layers) != 48
        or len(set(layers)) != 48
        or any(
            not isinstance(key, str) or re.fullmatch(r"0x[0-9a-f]+", key) is None
            for key in layers
        )
    ):
        mismatches.append("layer_keys")
    launches = payload.get("launches_observed")
    if isinstance(launches, bool) or not isinstance(launches, int) or launches < 48:
        mismatches.append("launches_observed")
    if mismatches:
        raise PassError("timing engagement contract mismatch: " + ",".join(mismatches))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    gate = commands.add_parser("validate")
    gate.add_argument("--gate", type=Path, required=True)
    gate.add_argument("--expected-gate-sha256", required=True)
    gate.add_argument("--candidate-source", type=Path, required=True)
    engagement = commands.add_parser("verify-engagement")
    engagement.add_argument("--engagement", type=Path, required=True)
    engagement.add_argument("--expected-gate-sha256", required=True)
    engagement.add_argument("--candidate-source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_gate(
            args.gate,
            expected_gate_sha256=args.expected_gate_sha256,
            candidate_source=args.candidate_source,
        )
    else:
        result = validate_engagement(
            args.engagement,
            gate_sha256=args.expected_gate_sha256,
            candidate_source=args.candidate_source,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
