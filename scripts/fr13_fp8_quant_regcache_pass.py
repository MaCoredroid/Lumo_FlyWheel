#!/usr/bin/env python3
"""Qualify and verify the fixed32 B1 FP8 quant regcache PASS sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


RECORD_SCHEMA = "fr13.fixed32.b1_fp8_quant_regcache.byte_ab.v1"
LIVE_SCHEMA = "fr13.fixed32.b1_fp8_quant_regcache.live_pass.v1"
SIDECAR_SCHEMA = "fr13.fixed32.b1_fp8_quant_regcache.production_pass.v1"
INSTANCE_ID = "astropy__astropy-12907"
TASK_MARKER = f"swe_verified:{INSTANCE_ID}"
SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
BLOCK_MAP_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
VLLM_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
HEX = frozenset("0123456789abcdef")
RECORD_KEYS = {
    "schema",
    "invocation",
    "target_forward_ordinal",
    "call_in_target_forward",
    "task_marker",
    "rows",
    "k",
    "group_size",
    "groups",
    "groups_per_cta",
    "ctas",
    "threads_per_cta",
    "output_bytes",
    "output_byte_equal",
    "output_mismatch_count",
    "output_first_mismatch",
    "scale_bytes",
    "scale_byte_equal",
    "scale_mismatch_count",
    "scale_first_mismatch",
    "scale_layout",
    "stock_served",
    "comparison_sampled",
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"non-finite JSON value: {value}")


def _regular(path: Path, label: str) -> bytes:
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_nlink != 1
    ):
        raise ValueError(f"{label} must be a single-link regular file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{label} is empty")
    return raw


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=_reject_nonfinite,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload, raw


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase commit")
    return value


def load_records(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = _regular(path, "FP8 quant byte records")
    if not raw.endswith(b"\n"):
        raise ValueError("FP8 quant byte records are not newline framed")
    records: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"blank FP8 quant byte record at line {number}")
        record = json.loads(
            line,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
        if not isinstance(record, dict):
            raise ValueError(f"FP8 quant byte record {number} is not an object")
        records.append(record)
    if len(records) < 128 or len(records) % 128 != 0:
        raise ValueError("FP8 quant gate did not cover complete 128-call target forwards")
    for invocation, record in enumerate(records):
        if set(record) != RECORD_KEYS:
            raise ValueError(f"FP8 quant record {invocation} key set drifted")
        expected = {
            "schema": RECORD_SCHEMA,
            "invocation": invocation,
            "target_forward_ordinal": invocation // 128,
            "call_in_target_forward": invocation % 128,
            "task_marker": TASK_MARKER,
            "rows": 32,
            "k": 5120,
            "group_size": 128,
            "groups": 1280,
            "groups_per_cta": 16,
            "ctas": 80,
            "threads_per_cta": 256,
            "output_bytes": 163840,
            "output_byte_equal": True,
            "output_mismatch_count": 0,
            "output_first_mismatch": None,
            "scale_bytes": 5120,
            "scale_byte_equal": True,
            "scale_mismatch_count": 0,
            "scale_first_mismatch": None,
            "scale_layout": "column_major_fp32_32x40_stride_1_32",
            "stock_served": True,
            "comparison_sampled": False,
        }
        if record != expected:
            raise ValueError(f"FP8 quant record {invocation} contract drifted")
    return records, raw


def _write_new(path: Path, raw: bytes, mode: int = 0o600) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    patch_source: Path,
    qualified_source_commit: str,
    out: Path,
) -> dict[str, Any]:
    from fr13_fp8_quant_regcache_runtime import validate_binary

    expected_live_sha256 = _sha256(expected_live_sha256, "live result")
    expected_candidate_sha256 = _sha256(
        expected_candidate_sha256, "candidate binary"
    )
    qualified_source_commit = _commit(
        qualified_source_commit, "qualified source"
    )
    candidate = validate_binary(candidate_so, expected_candidate_sha256)
    live, live_raw = load_json(live_result, "FP8 quant live result")
    if _digest(live_raw) != expected_live_sha256:
        raise ValueError("FP8 quant live-result SHA-256 mismatch")
    if (
        live.get("schema") != LIVE_SCHEMA
        or live.get("status") != "PASS"
        or live.get("candidate_sha256") != expected_candidate_sha256
        or live.get("source_commit") != qualified_source_commit
        or live.get("task_ids") != [INSTANCE_ID]
        or live.get("output_mismatching_bytes") != 0
        or live.get("scale_mismatching_bytes") != 0
        or live.get("stock_served") is not True
        or live.get("comparison_sampled") is not False
    ):
        raise ValueError("FP8 quant live result is not a production-qualified PASS")
    patch_raw = _regular(patch_source, "FP8 quant patch source")
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "candidate_sha256": expected_candidate_sha256,
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": _digest(patch_raw),
        "qualified_source_commit": qualified_source_commit,
        "vllm_commit": VLLM_COMMIT,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest(canonical_bytes(live)),
        "qualification_task_id": INSTANCE_ID,
        "qualified_invocations": live["comparisons"],
        "qualified_target_forwards": live["target_forwards"],
        "required_runtime": "Hydra27 fixed32 physical32 K64 root1 B1",
        "production_scope": "BF16[32,5120] group128 to FP8 E4M3 plus FP32 scales",
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    _write_new(out, canonical_bytes(sidecar) + b"\n")
    return sidecar


def verify_sidecar(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    patch_source: Path,
) -> dict[str, Any]:
    from fr13_fp8_quant_regcache_runtime import validate_binary

    expected_sidecar_sha256 = _sha256(
        expected_sidecar_sha256, "production sidecar"
    )
    expected_candidate_sha256 = _sha256(
        expected_candidate_sha256, "candidate binary"
    )
    candidate = validate_binary(candidate_so, expected_candidate_sha256)
    payload, raw = load_json(sidecar_path, "FP8 quant production sidecar")
    if _digest(raw) != expected_sidecar_sha256:
        raise ValueError("FP8 quant production-sidecar SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise ValueError("FP8 quant sidecar canonical digest mismatch")
    expected_keys = {
        "schema",
        "status",
        "candidate_sha256",
        "candidate_bytes",
        "patch_source_sha256",
        "qualified_source_commit",
        "vllm_commit",
        "live_result_sha256",
        "live_result_canonical_sha256",
        "qualification_task_id",
        "qualified_invocations",
        "qualified_target_forwards",
        "required_runtime",
        "production_scope",
    }
    patch_raw = _regular(patch_source, "FP8 quant patch source")
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("candidate_sha256") != expected_candidate_sha256
        or payload.get("candidate_bytes") != candidate["bytes"]
        or payload.get("patch_source_sha256") != _digest(patch_raw)
        or payload.get("vllm_commit") != VLLM_COMMIT
        or payload.get("qualification_task_id") != INSTANCE_ID
        or type(payload.get("qualified_invocations")) is not int
        or payload["qualified_invocations"] < 128
        or payload["qualified_invocations"] % 128 != 0
        or payload.get("qualified_target_forwards")
        != payload["qualified_invocations"] // 128
        or payload.get("required_runtime")
        != "Hydra27 fixed32 physical32 K64 root1 B1"
        or payload.get("production_scope")
        != "BF16[32,5120] group128 to FP8 E4M3 plus FP32 scales"
    ):
        raise ValueError("FP8 quant production sidecar contract drifted")
    _commit(payload.get("qualified_source_commit"), "qualified source")
    _sha256(payload.get("live_result_sha256"), "live result")
    _sha256(payload.get("live_result_canonical_sha256"), "canonical live result")
    payload["canonical_sha256"] = canonical
    return payload


def qualify(
    *,
    records_path: Path,
    binary_attestation_path: Path,
    task_arm_path: Path,
    diagnostic_path: Path,
    container_env_path: Path,
    terminal_path: Path,
    traffic_path: Path,
    runtime_manifest_launch: Path,
    runtime_manifest_end: Path,
    external_manifest_launch: Path,
    external_manifest_end: Path,
    candidate_so: Path,
    expected_candidate_sha256: str,
    patch_source: Path,
    source_commit: str,
    out_live: Path,
    out_sidecar: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from fr13_fp8_quant_regcache_runtime import BINARY_SCHEMA, validate_binary

    expected_candidate_sha256 = _sha256(
        expected_candidate_sha256, "candidate binary"
    )
    source_commit = _commit(source_commit, "qualification source")
    candidate = validate_binary(candidate_so, expected_candidate_sha256)
    records, records_raw = load_records(records_path)
    binary, binary_raw = load_json(binary_attestation_path, "binary attestation")
    task_arm, task_arm_raw = load_json(task_arm_path, "real-task arm")
    diagnostic, diagnostic_raw = load_json(diagnostic_path, "B1 diagnostic")
    terminal, terminal_raw = load_json(terminal_path, "eager terminal")
    traffic, traffic_raw = load_json(traffic_path, "eager traffic audit")
    container_env_raw = _regular(container_env_path, "container environment")
    runtime_launch_raw = _regular(runtime_manifest_launch, "launch runtime manifest")
    runtime_end_raw = _regular(runtime_manifest_end, "end runtime manifest")
    external_launch_raw = _regular(external_manifest_launch, "launch external manifest")
    external_end_raw = _regular(external_manifest_end, "end external manifest")
    patch_raw = _regular(patch_source, "FP8 quant patch source")
    if runtime_launch_raw != runtime_end_raw:
        raise ValueError("runtime manifest drifted during FP8 quant gate")
    if external_launch_raw != external_end_raw:
        raise ValueError("external manifest drifted during FP8 quant gate")
    if (
        binary.get("schema") != BINARY_SCHEMA
        or binary.get("status") != "INSTALLED"
        or binary.get("selector") != "byte_ab"
        or binary.get("diagnostic_enabled") is not True
        or binary.get("production_enabled") is not False
        or binary.get("candidate_sha256") != expected_candidate_sha256
        or binary.get("candidate_bytes") != candidate["bytes"]
        or binary.get("smoke_load_passed") is not True
        or binary.get("source_commit") != source_commit
        or binary.get("patch_source_sha256") != _digest(patch_raw)
        or binary.get("vllm_commit") != VLLM_COMMIT
        or binary.get("production_sidecar_sha256") is not None
    ):
        raise ValueError("FP8 quant binary attestation drifted")
    expected_arm = {
        "schema": "fr13-fixed32-cutlass-streamk-real-task-arm-v1",
        "state": "ended",
        "instance_id": INSTANCE_ID,
        "marker": TASK_MARKER,
    }
    if any(task_arm.get(key) != value for key, value in expected_arm.items()):
        raise ValueError("FP8 quant real-task arm drifted")
    if (
        diagnostic.get("schema") != "fr13-fixed32-b1-diagnostic-v1"
        or diagnostic.get("run_classification") != "b1_diagnostic"
        or diagnostic.get("gate_eligible") is not False
        or diagnostic.get("floor_acceptance_eligible") is not False
        or diagnostic.get("max_num_seqs") != 1
        or diagnostic.get("swe_concurrency") != 1
        or diagnostic.get("subset_sha256") != SUBSET_SHA256
        or diagnostic.get("task_ids") != [INSTANCE_ID]
    ):
        raise ValueError("FP8 quant B1 diagnostic binding drifted")
    if terminal != {
        "schema": "fr13-fixed32-eager-kernel-terminal-v1",
        "run_classification": "eager_kernel_byte_diagnostic",
        "acceptance_valid": False,
        "flush_protocol_used": False,
    }:
        raise ValueError("FP8 quant eager terminal binding drifted")
    if (
        traffic.get("schema")
        != "fr13-fixed32-eager-kernel-traffic-audit-skip-v1"
        or traffic.get("run_classification") != "eager_kernel_byte_diagnostic"
        or traffic.get("acceptance_valid") is not False
        or traffic.get("authenticated_engine_ledger_snapshotted") is not True
        or traffic.get("graph_census_audit_used") is not False
    ):
        raise ValueError("FP8 quant eager traffic binding drifted")
    environment = container_env_raw.decode("ascii").splitlines()
    expected_environment = (
        "FR13_FIXED32_B1_FP8_QUANT_REGCACHE=byte_ab",
        f"FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO_SHA256={expected_candidate_sha256}",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json",
        "FR13_FIXED32_CUTLASS_WAVE=stock",
        "ENFORCE_EAGER=1",
    )
    for expected in expected_environment:
        if environment.count(expected) != 1:
            raise ValueError(f"FP8 quant environment pin drifted: {expected}")

    live = {
        "schema": LIVE_SCHEMA,
        "status": "PASS",
        "run_classification": "one_real_swe_verified_k64_root_b1_fp8_quant_byte_diagnostic",
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "task_ids": [INSTANCE_ID],
        "task_marker": TASK_MARKER,
        "batch_size": 1,
        "concurrency": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "candidate": "fp8_quant_regcache_r32k5120_group128",
        "candidate_sha256": expected_candidate_sha256,
        "candidate_bytes": candidate["bytes"],
        "source_commit": source_commit,
        "patch_source_sha256": _digest(patch_raw),
        "vllm_commit": VLLM_COMMIT,
        "comparisons": len(records),
        "target_forwards": len(records) // 128,
        "output_mismatching_bytes": 0,
        "scale_mismatching_bytes": 0,
        "stock_served": True,
        "comparison_sampled": False,
        "records_sha256": _digest(records_raw),
        "binary_attestation_sha256": _digest(binary_raw),
        "real_task_arm_sha256": _digest(task_arm_raw),
        "diagnostic_binding_sha256": _digest(diagnostic_raw),
        "container_env_sha256": _digest(container_env_raw),
        "terminal_sha256": _digest(terminal_raw),
        "traffic_audit_sha256": _digest(traffic_raw),
        "runtime_manifest_sha256": _digest(runtime_launch_raw),
        "external_manifest_sha256": _digest(external_launch_raw),
    }
    _write_new(out_live, canonical_bytes(live) + b"\n")
    sidecar = issue_sidecar(
        live_result=out_live,
        expected_live_sha256=_digest(out_live.read_bytes()),
        candidate_so=candidate_so,
        expected_candidate_sha256=expected_candidate_sha256,
        patch_source=patch_source,
        qualified_source_commit=source_commit,
        out=out_sidecar,
    )
    return live, sidecar


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    qualify_parser = subparsers.add_parser("qualify")
    for name in (
        "records",
        "binary-attestation",
        "task-arm",
        "diagnostic",
        "container-env",
        "terminal",
        "traffic",
        "runtime-manifest-launch",
        "runtime-manifest-end",
        "external-manifest-launch",
        "external-manifest-end",
        "candidate-so",
        "patch-source",
        "out-live",
        "out-sidecar",
    ):
        qualify_parser.add_argument(f"--{name}", type=Path, required=True)
    qualify_parser.add_argument("--expected-candidate-sha256", required=True)
    qualify_parser.add_argument("--source-commit", required=True)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--sidecar", type=Path, required=True)
    verify_parser.add_argument("--expected-sidecar-sha256", required=True)
    verify_parser.add_argument("--candidate-so", type=Path, required=True)
    verify_parser.add_argument("--expected-candidate-sha256", required=True)
    verify_parser.add_argument("--patch-source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "qualify":
        live, sidecar = qualify(
            records_path=args.records,
            binary_attestation_path=args.binary_attestation,
            task_arm_path=args.task_arm,
            diagnostic_path=args.diagnostic,
            container_env_path=args.container_env,
            terminal_path=args.terminal,
            traffic_path=args.traffic,
            runtime_manifest_launch=args.runtime_manifest_launch,
            runtime_manifest_end=args.runtime_manifest_end,
            external_manifest_launch=args.external_manifest_launch,
            external_manifest_end=args.external_manifest_end,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            patch_source=args.patch_source,
            source_commit=args.source_commit,
            out_live=args.out_live,
            out_sidecar=args.out_sidecar,
        )
        result = {"live": live, "sidecar": sidecar}
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            patch_source=args.patch_source,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
