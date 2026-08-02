#!/usr/bin/env python3
"""Validate SFWD exact4 B4 bytes and bind the future B1/B4 prerequisite."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_state_fusion_v1"
B4_LIVE_SCHEMA = "fr13.fixed32.sfwd_state_fusion.exact4_b4_live_gate.v1"
B4_QUALIFICATION_SCHEMA = "fr13.fixed32.sfwd_state_fusion.exact4_b4_qualification.v1"
B1_LIVE_SCHEMA = "fr13.fixed32.sfwd_state_fusion.live_pass.v1"
PREREQUISITE_SCHEMA = "fr13.fixed32.sfwd_state_fusion.b1_b4_production_prerequisite.v1"
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXPECTED_TASK_MARKERS = tuple(
    f"swe_verified:{task_id}" for task_id in EXPECTED_TASK_IDS
)
EXPECTED_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
B1_TASK_MARKER = "swe_verified:astropy__astropy-12907"
CANDIDATE_KERNEL_FUNCTION = "_fr13_fixed32_sfwd_state_fusion_kernel"


class QualificationError(ValueError):
    """A byte result or prerequisite is incomplete or inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str, *, max_bytes: int) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise QualificationError(f"{label} does not exist: {path}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > max_bytes
    ):
        raise QualificationError(f"{label} is not a bounded regular file: {path}")
    return info


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise QualificationError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise QualificationError(f"non-finite JSON value: {value}")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular_file(path, label, max_bytes=1024 * 1024)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not strict ASCII JSON") from error
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must contain a JSON object")
    return payload, raw


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QualificationError(f"{label} is not a lowercase SHA-256")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def candidate_kernel_ast_sha256(path: Path) -> str:
    _regular_file(path, "SFWD kernel source", max_bytes=4 * 1024 * 1024)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError) as error:
        raise QualificationError("SFWD kernel source cannot be parsed") from error
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == CANDIDATE_KERNEL_FUNCTION
    ]
    if len(nodes) != 1:
        raise QualificationError("SFWD candidate kernel function is not unique")
    encoded = ast.dump(nodes[0], include_attributes=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_b4_live_result(
    live_result: Path,
    *,
    expected_live_sha256: str,
    kernel_source: Path,
    patcher_source: Path,
    expected_source_commit: str | None = None,
) -> dict[str, Any]:
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected B4 live-result SHA-256"
    )
    payload, raw = _read_json(live_result, "SFWD exact4 B4 live PASS")
    live_sha256 = hashlib.sha256(raw).hexdigest()
    if live_sha256 != expected_live_sha256:
        raise QualificationError("SFWD exact4 B4 live PASS SHA-256 mismatch")
    kernel_sha256 = sha256_file(kernel_source)
    patcher_sha256 = sha256_file(patcher_source)
    expected: dict[str, object] = {
        "schema": B4_LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "candidate": CANDIDATE,
        "task_set": "canonical real SWE-Verified exact4 B4",
        "task_count": 4,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_markers": list(EXPECTED_TASK_MARKERS),
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "real_task_authenticated": True,
        "batch_size": 4,
        "concurrency": 4,
        "physical_rows_per_request": 32,
        "physical_rows_total": 128,
        "layer_count": 48,
        "comparison_records": 48,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "mismatching_records": 0,
        "differing_bytes": 0,
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "candidate_shadow_only": True,
        "served_result": "reference",
        "reference_always_served": True,
        "probe_inputs": False,
        "synthetic_inputs": False,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "production_eligible": False,
        "kernel_source_sha256": kernel_sha256,
        "patcher_source_sha256": patcher_sha256,
        "errors": [],
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise QualificationError(
                f"SFWD exact4 B4 live PASS {key} mismatch: "
                f"{payload.get(key)!r} != {expected_value!r}"
            )
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("SFWD exact4 B4 source commit is invalid")
    if expected_source_commit is not None and source_commit != expected_source_commit:
        raise QualificationError("SFWD exact4 B4 source commit is stale")
    layer_keys = payload.get("layer_keys")
    if (
        not isinstance(layer_keys, list)
        or len(layer_keys) != 48
        or len(set(layer_keys)) != 48
        or any(
            not isinstance(key, str) or re.fullmatch(r"0x[0-9a-f]+", key) is None
            for key in layer_keys
        )
    ):
        raise QualificationError("SFWD exact4 B4 does not cover 48 unique layers")
    for field in (
        "engine_ledger_chain_head_sha256",
        "real_task_arm_sha256",
        "runtime_manifest_sha256",
        "runner_sha256",
    ):
        _require_sha256(payload.get(field), field)
    return {
        "schema": B4_QUALIFICATION_SCHEMA,
        "status": "QUALIFIED_BYTE_ONLY",
        "candidate": CANDIDATE,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_task_markers": list(EXPECTED_TASK_MARKERS),
        "qualification_subset_sha256": EXPECTED_SUBSET_SHA256,
        "qualification_batch_size": 4,
        "qualification_concurrency": 4,
        "qualification_physical_rows_per_request": 32,
        "qualification_layer_count": 48,
        "live_result_sha256": live_sha256,
        "kernel_source_sha256": kernel_sha256,
        "patcher_source_sha256": patcher_sha256,
        "candidate_kernel_ast_sha256": candidate_kernel_ast_sha256(kernel_source),
        "qualification_source_commit": source_commit,
        "served_result_during_qualification": "reference",
        "candidate_shadow_only": True,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "candidate_serving_permitted": False,
        "remaining_prerequisite": "authenticated source-equivalent B1 byte PASS",
    }


def validate_b1_live_result(
    live_result: Path,
    *,
    expected_live_sha256: str,
    kernel_source: Path,
) -> dict[str, Any]:
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected B1 live-result SHA-256"
    )
    payload, raw = _read_json(live_result, "SFWD B1 live PASS")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_live_sha256:
        raise QualificationError("SFWD B1 live PASS SHA-256 mismatch")
    expected = {
        "schema": B1_LIVE_SCHEMA,
        "status": "byte_pass_source_only",
        "run_classification": (
            "one_real_swe_verified_full_vocab_b1_byte_timing_diagnostic"
        ),
        "candidate": CANDIDATE,
        "source_sha256": sha256_file(kernel_source),
        "task_marker": B1_TASK_MARKER,
        "batch": 1,
        "layer_count": 48,
        "physical_rows_per_request": 32,
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise QualificationError(f"SFWD B1 live PASS {key} mismatch")
    layers = payload.get("layer_keys")
    if not isinstance(layers, list) or len(layers) != 48 or len(set(layers)) != 48:
        raise QualificationError("SFWD B1 live PASS lacks 48 unique layers")
    return {
        "live_result_sha256": actual_sha256,
        "kernel_source_sha256": expected["source_sha256"],
        "candidate_kernel_ast_sha256": candidate_kernel_ast_sha256(kernel_source),
    }


def bind_prerequisites(
    *,
    b1_live_result: Path,
    expected_b1_sha256: str,
    b1_kernel_source: Path,
    b4_qualification: Path,
    expected_b4_qualification_sha256: str,
    b4_kernel_source: Path,
) -> dict[str, Any]:
    b1 = validate_b1_live_result(
        b1_live_result,
        expected_live_sha256=expected_b1_sha256,
        kernel_source=b1_kernel_source,
    )
    expected_b4_qualification_sha256 = _require_sha256(
        expected_b4_qualification_sha256, "expected B4 qualification SHA-256"
    )
    b4, b4_raw = _read_json(b4_qualification, "SFWD B4 qualification")
    actual_b4_sha256 = hashlib.sha256(b4_raw).hexdigest()
    if actual_b4_sha256 != expected_b4_qualification_sha256:
        raise QualificationError("SFWD B4 qualification SHA-256 mismatch")
    required_b4 = {
        "schema": B4_QUALIFICATION_SCHEMA,
        "status": "QUALIFIED_BYTE_ONLY",
        "candidate": CANDIDATE,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_batch_size": 4,
        "qualification_concurrency": 4,
        "qualification_physical_rows_per_request": 32,
        "qualification_layer_count": 48,
        "served_result_during_qualification": "reference",
        "candidate_shadow_only": True,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "candidate_serving_permitted": False,
    }
    for key, expected_value in required_b4.items():
        if b4.get(key) != expected_value:
            raise QualificationError(f"SFWD B4 qualification {key} mismatch")
    b4_source_sha256 = sha256_file(b4_kernel_source)
    if b4.get("kernel_source_sha256") != b4_source_sha256:
        raise QualificationError("SFWD B4 qualification kernel source is stale")
    b4_ast_sha256 = candidate_kernel_ast_sha256(b4_kernel_source)
    if b4.get("candidate_kernel_ast_sha256") != b4_ast_sha256:
        raise QualificationError("SFWD B4 qualification candidate kernel is stale")
    if b1["candidate_kernel_ast_sha256"] != b4_ast_sha256:
        raise QualificationError("SFWD B1/B4 candidate kernel implementations differ")
    return {
        "schema": PREREQUISITE_SCHEMA,
        "status": "PREREQUISITES_SATISFIED_BYTE_ONLY",
        "candidate": CANDIDATE,
        "b1_live_result_sha256": b1["live_result_sha256"],
        "b1_kernel_source_sha256": b1["kernel_source_sha256"],
        "b4_qualification_sha256": actual_b4_sha256,
        "b4_kernel_source_sha256": b4_source_sha256,
        "candidate_kernel_ast_sha256": b4_ast_sha256,
        "b1_batch_size": 1,
        "b4_batch_size": 4,
        "b4_concurrency": 4,
        "b4_task_ids": list(EXPECTED_TASK_IDS),
        "byte_prerequisites_satisfied": True,
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "candidate_serving_permitted": False,
        "note": "binding is a production prerequisite, not a serving credential",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    issue = commands.add_parser("issue-b4")
    issue.add_argument("--live-result", type=Path, required=True)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--kernel-source", type=Path, required=True)
    issue.add_argument("--patcher-source", type=Path, required=True)
    issue.add_argument("--expected-source-commit")
    issue.add_argument("--out", type=Path, required=True)
    bind = commands.add_parser("bind-prerequisites")
    bind.add_argument("--b1-live-result", type=Path, required=True)
    bind.add_argument("--expected-b1-sha256", required=True)
    bind.add_argument("--b1-kernel-source", type=Path, required=True)
    bind.add_argument("--b4-qualification", type=Path, required=True)
    bind.add_argument("--expected-b4-qualification-sha256", required=True)
    bind.add_argument("--b4-kernel-source", type=Path, required=True)
    bind.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "issue-b4":
        result = validate_b4_live_result(
            args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            kernel_source=args.kernel_source,
            patcher_source=args.patcher_source,
            expected_source_commit=args.expected_source_commit,
        )
    else:
        result = bind_prerequisites(
            b1_live_result=args.b1_live_result,
            expected_b1_sha256=args.expected_b1_sha256,
            b1_kernel_source=args.b1_kernel_source,
            b4_qualification=args.b4_qualification,
            expected_b4_qualification_sha256=(args.expected_b4_qualification_sha256),
            b4_kernel_source=args.b4_kernel_source,
        )
    _write_json(args.out, result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
