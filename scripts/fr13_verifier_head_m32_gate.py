#!/usr/bin/env python3
"""Validate the authenticated B1 raw-BF16 verifier-head M32 shadow gate."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

terminal = importlib.import_module("fr13_draft_head_m32_pass")


LIVE_SCHEMA = "fr13.fixed32.verifier_head_m32_shadow.v1"
GATE_SCHEMA = "fr13.fixed32.verifier_head_m32_real_b1_gate.v1"
EXPECTED_INSTANCE = "astropy__astropy-12907"
EXPECTED_SO_SHA256 = "5b5e8c3051f29bc4f65ef93c96ed22ef38ef07a1754e9c36a167e5158f71f4b7"
EXPECTED_SO_BYTES = 186048
EXPECTED_KERNEL_SHA256 = (
    "7cbc9f5157d8e93ee35930b028d97d0c3b1a26a9d79aa87ec6061928f8161768"
)
EXPECTED_BUILD_ATTESTATION_SHA256 = (
    "780ea833962806ea4a374c3092c33ad75f2d23fd255daabb6d39f69533fc3d5c"
)
EXPECTED_ELEMENTS = 32 * 248320
EXPECTED_BYTES = EXPECTED_ELEMENTS * 2
EXPECTED_TOPOLOGY = {
    "mode": "hydra27_fixed32",
    "batch_size": 1,
    "physical_rows": 32,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "enforce_eager": True,
}
EXPECTED_GEOMETRY = {
    "hidden_shape": [32, 5120],
    "weight_shape": [248320, 5120],
    "output_shape": [32, 248320],
    "dtype": "torch.bfloat16",
}
LIVE_KEYS = frozenset(
    {
        "schema",
        "status",
        "suite",
        "instance_id",
        "source_commit",
        "patch_source_sha256",
        "candidate_so_sha256",
        "kernel_source_sha256",
        "task_marker",
        "topology",
        "geometry",
        "comparison_calls",
        "compared_elements",
        "compared_bytes",
        "raw_bf16_mismatches",
        "reference_preservation_mismatches",
        "reference_sha256",
        "candidate_sha256",
        "reference_always_served",
        "candidate_returned",
        "served_return",
        "performance_measurement",
        "timing_eligible",
    }
)


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in terminal.HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in terminal.HEX for character in value)
    ):
        raise ValueError(f"{label} is not a lowercase source commit")
    return value


def validate_live_result(
    payload: dict[str, Any],
    *,
    expected_source_commit: str,
    expected_patch_sha256: str,
) -> dict[str, Any]:
    expected_source_commit = _commit(expected_source_commit, "expected source")
    expected_patch_sha256 = _sha256(expected_patch_sha256, "expected patch source")
    if frozenset(payload) != LIVE_KEYS:
        raise ValueError("verifier-head live result key set drifted")
    for label in ("reference_sha256", "candidate_sha256"):
        _sha256(payload.get(label), label)
    if (
        payload.get("schema") != LIVE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("source_commit") != expected_source_commit
        or payload.get("patch_source_sha256") != expected_patch_sha256
        or payload.get("candidate_so_sha256") != EXPECTED_SO_SHA256
        or payload.get("kernel_source_sha256") != EXPECTED_KERNEL_SHA256
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload.get("topology") != EXPECTED_TOPOLOGY
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("comparison_calls") != 1
        or payload.get("compared_elements") != EXPECTED_ELEMENTS
        or payload.get("compared_bytes") != EXPECTED_BYTES
        or payload.get("raw_bf16_mismatches") != 0
        or payload.get("reference_preservation_mismatches") != 0
        or payload.get("reference_sha256") != payload.get("candidate_sha256")
        or payload.get("reference_always_served") is not True
        or payload.get("candidate_returned") is not False
        or payload.get("served_return") != "incumbent BF16 logits object unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("timing_eligible") is not False
    ):
        raise ValueError("verifier-head live PASS contract drifted")
    return payload


def _validate_binary_and_sources(
    *,
    candidate_so: Path,
    kernel_source: Path,
    patch_source: Path,
    build_attestation: Path,
    live: dict[str, Any],
) -> dict[str, str | int]:
    for path, label in (
        (candidate_so, "candidate SO"),
        (kernel_source, "kernel source"),
        (patch_source, "patch source"),
        (build_attestation, "build attestation"),
    ):
        terminal.require_regular_file(path, label)
    attestation, _ = terminal.load_json(build_attestation)
    binary = attestation.get("binary")
    source = attestation.get("source")
    if (
        candidate_so.stat().st_size != EXPECTED_SO_BYTES
        or terminal.sha256_file(candidate_so) != EXPECTED_SO_SHA256
        or terminal.sha256_file(kernel_source) != EXPECTED_KERNEL_SHA256
        or terminal.sha256_file(patch_source) != live["patch_source_sha256"]
        or terminal.sha256_file(build_attestation)
        != EXPECTED_BUILD_ATTESTATION_SHA256
        or attestation.get("schema")
        != "fr13.fixed32.bf16_verifier_head_m32_sm121a_build.v1"
        or attestation.get("status") != "BUILT_UNQUALIFIED"
        or attestation.get("production_default_enabled") is not False
        or attestation.get("real_task_correctness") is not False
        or attestation.get("byte_equality_claim") is not False
        or not isinstance(binary, dict)
        or binary.get("sha256") != EXPECTED_SO_SHA256
        or binary.get("bytes") != EXPECTED_SO_BYTES
        or not isinstance(source, dict)
        or source.get("path") != "csrc/fr13_bf16_verifier_head_m32_sm121a.cu"
        or source.get("sha256") != EXPECTED_KERNEL_SHA256
    ):
        raise ValueError("verifier-head build, binary, or source identity drifted")
    return {
        "candidate_so_sha256": EXPECTED_SO_SHA256,
        "candidate_so_bytes": EXPECTED_SO_BYTES,
        "kernel_source_sha256": EXPECTED_KERNEL_SHA256,
        "patch_source_sha256": live["patch_source_sha256"],
        "build_attestation_sha256": EXPECTED_BUILD_ATTESTATION_SHA256,
    }


def _validate_terminal_evidence(
    *,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    repo: Path,
) -> dict[str, Any]:
    audit, _ = terminal.load_json(chat_traffic_audit)
    stream = audit.get("complete_stream")
    events = (
        stream.get("complete_work_census_events") if isinstance(stream, dict) else None
    )
    if type(events) is not int or events < 1:
        raise ValueError("verifier-head traffic audit has no complete events")
    traffic = terminal.validate_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        expected_events=events,
    )
    terminal.validate_rebuilt_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        repo=repo,
    )
    task = audit["tasks"][EXPECTED_INSTANCE]
    if task["terminal"]["eval"] != {
        "verdict": "resolved",
        "passed": True,
        "harness_exit_code": 0,
    }:
        raise ValueError("verifier-head real SWE-Verified task did not resolve")

    flush, _ = terminal.load_json(final_flush)
    boundary, _ = terminal.load_json(boundary_snapshot)
    ack = flush.get("ack")
    fixed32 = (
        boundary.get("metrics", {}).get("fixed32")
        if isinstance(boundary.get("metrics"), dict)
        else None
    )
    if not isinstance(ack, dict) or not isinstance(fixed32, dict):
        raise ValueError("verifier-head terminal evidence is malformed")
    synthetic_live = {
        "completed_events": events,
        "flush_generation": ack.get("generation"),
        "flush_nonce": ack.get("nonce"),
        "producer_pid": ack.get("producer_pid"),
        "events_sha256": fixed32.get("events_sha256"),
        "boundary_snapshot_sha256": terminal.sha256_file(boundary_snapshot),
    }
    terminal_evidence = terminal.validate_live_evidence(
        live_payload=synthetic_live,
        final_flush_path=final_flush,
        boundary_snapshot_path=boundary_snapshot,
    )
    return {**traffic, **terminal_evidence}


def validate_gate(
    *,
    live_result: Path,
    candidate_so: Path,
    kernel_source: Path,
    patch_source: Path,
    build_attestation: Path,
    expected_source_commit: str,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    repo: Path,
) -> dict[str, Any]:
    live, _ = terminal.load_json(live_result)
    patch_sha256 = terminal.sha256_file(patch_source)
    validate_live_result(
        live,
        expected_source_commit=expected_source_commit,
        expected_patch_sha256=patch_sha256,
    )
    identities = _validate_binary_and_sources(
        candidate_so=candidate_so,
        kernel_source=kernel_source,
        patch_source=patch_source,
        build_attestation=build_attestation,
        live=live,
    )
    evidence = _validate_terminal_evidence(
        final_flush=final_flush,
        boundary_snapshot=boundary_snapshot,
        chat_traffic_audit=chat_traffic_audit,
        repo=repo,
    )
    return {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "source_commit": expected_source_commit,
        "candidate": identities,
        "live_result_sha256": terminal.sha256_file(live_result),
        "comparison_calls": 1,
        "compared_elements": EXPECTED_ELEMENTS,
        "compared_bytes": EXPECTED_BYTES,
        "raw_bf16_mismatches": 0,
        "reference_preservation_mismatches": 0,
        "reference_always_served": True,
        "candidate_returned": False,
        "task_resolved": True,
        "completed_events": evidence["completed_events"],
        "final_flush_sha256": terminal.sha256_file(final_flush),
        "boundary_snapshot_sha256": terminal.sha256_file(boundary_snapshot),
        "chat_traffic_audit_sha256": terminal.sha256_file(chat_traffic_audit),
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(terminal.canonical_bytes(payload) + b"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-result", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--patch-source", type=Path, required=True)
    parser.add_argument("--build-attestation", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_gate(
        live_result=args.live_result,
        candidate_so=args.candidate_so,
        kernel_source=args.kernel_source,
        patch_source=args.patch_source,
        build_attestation=args.build_attestation,
        expected_source_commit=args.expected_source_commit,
        final_flush=args.final_flush,
        boundary_snapshot=args.boundary_snapshot,
        chat_traffic_audit=args.chat_traffic_audit,
        repo=args.repo,
    )
    _write(args.out, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
