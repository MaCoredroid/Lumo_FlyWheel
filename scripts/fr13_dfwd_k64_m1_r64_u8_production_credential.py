#!/usr/bin/env python3
"""Issue and verify the exact-B1 K64/root1 DFWD U8 production credential."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any


SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

gate = importlib.import_module("fr13_dfwd_k64_m1_r64_u8_gate")
terminal = gate.terminal


CREDENTIAL_SCHEMA = "fr13.fixed32.dfwd_k64_m1_r64_u8_production_credential.v1"
VALIDATION_SCHEMA = "fr13.fixed32.dfwd_k64_m1_r64_u8_production_validation.v1"
ENGAGEMENT_SCHEMA = "fr13.fixed32.dfwd_k64_m1_r64_u8_production_engagement.v1"
SELECTOR = "fr13_bf16_k64_m1_r64_u8_direct"
GRAPH_SIGNATURE = "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
INPUT_KEYS = frozenset(
    {
        "build_attestation_sha256",
        "candidate_so_bytes",
        "candidate_so_sha256",
        "candidate_source_sha256",
        "fa2_sha256",
        "patch_source_sha256",
        "runner_sha256",
        "subset_sha256",
        "vocab_blocks_sha256",
    }
)
CREDENTIAL_KEYS = frozenset(
    {
        "candidate",
        "candidate_returned_during_qualification",
        "captured_mtp_depths",
        "comparison_scope",
        "evidence_sha256",
        "floor_acceptance_eligible",
        "geometry",
        "graph_contract",
        "incumbent_served_during_qualification",
        "inputs",
        "performance_claim",
        "production_default_enabled",
        "qualification_schema",
        "qualification_task_id",
        "raw_bf16_mismatches",
        "schema",
        "selector",
        "serve_policy",
        "source_commit",
        "status",
        "suite",
        "task_marker",
        "timing_eligible",
        "topology",
    }
)
EVIDENCE_KEYS = frozenset(
    {
        "boundary_snapshot_sha256",
        "chat_traffic_audit_sha256",
        "events_sha256",
        "final_flush_sha256",
        "live_result_sha256",
    }
)
GRAPH_CONTRACT = {
    "batch_size": 1,
    "captured_loop_calls": 4,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "execution_basis": "cudagraph_replay",
    "graph_signature": GRAPH_SIGNATURE,
    "mode": "hydra27_fixed32",
    "root_calls": 1,
}
ENGAGEMENT_KEYS = frozenset(
    {
        "candidate_served",
        "candidate_so_sha256",
        "candidate_source_sha256",
        "capture_origin",
        "captured_loop_calls",
        "drafter_graph_id",
        "drafter_graph_signature",
        "execution_basis",
        "fallback_calls",
        "forward_step_index",
        "geometry",
        "incumbent_head_calls",
        "observed_measured_replays_at_least",
        "performance_claim",
        "production_credential_sha256",
        "qualification_candidate",
        "runtime_mode",
        "schema",
        "selected_root_calls",
        "selector",
        "source_commit",
        "status",
        "steady_state_synchronizations",
    }
)


def _exact_dict(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ValueError(f"{label} key set drifted")
    return value


def _regular(path: Path, label: str) -> None:
    terminal.require_regular_file(path, label)


def _canonical_load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = terminal.load_json(path)
    if raw != terminal.canonical_bytes(payload) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return payload, raw


def _credential_from_gate(validated: dict[str, Any]) -> dict[str, Any]:
    if (
        validated.get("schema") != gate.GATE_SCHEMA
        or validated.get("status") != "PASS"
        or validated.get("captured_mtp_depths") != [1, 2, 3, 4]
        or validated.get("raw_bf16_mismatches") != 0
        or validated.get("reference_always_served") is not True
        or validated.get("candidate_returned") is not False
        or validated.get("task_resolved") is not True
        or validated.get("performance_measurement") is not False
        or validated.get("timing_eligible") is not False
        or validated.get("production_eligible") is not False
    ):
        raise ValueError("DFWD U8 gate result is not an exact shadow PASS")
    inputs = _exact_dict(validated.get("inputs"), INPUT_KEYS, "gate inputs")
    evidence = {
        "boundary_snapshot_sha256": validated["boundary_snapshot_sha256"],
        "chat_traffic_audit_sha256": validated["chat_traffic_audit_sha256"],
        "events_sha256": validated["events_sha256"],
        "final_flush_sha256": validated["final_flush_sha256"],
        "live_result_sha256": validated["live_result_sha256"],
    }
    for key, value in evidence.items():
        gate._sha256(value, f"credential evidence {key}")
    return {
        "schema": CREDENTIAL_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "qualification_schema": gate.GATE_SCHEMA,
        "qualification_task_id": gate.EXPECTED_INSTANCE,
        "task_marker": f"swe_verified:{gate.EXPECTED_INSTANCE}",
        "source_commit": validated["source_commit"],
        "selector": SELECTOR,
        "serve_policy": "candidate_only_after_internal_attestation",
        "topology": validated["topology"],
        "geometry": validated["geometry"],
        "candidate": validated["candidate"],
        "inputs": inputs,
        "evidence_sha256": evidence,
        "captured_mtp_depths": [1, 2, 3, 4],
        "comparison_scope": gate.COMPARISON_SCOPE,
        "raw_bf16_mismatches": 0,
        "incumbent_served_during_qualification": True,
        "candidate_returned_during_qualification": False,
        "graph_contract": GRAPH_CONTRACT,
        "production_default_enabled": False,
        "timing_eligible": True,
        "performance_claim": False,
        "floor_acceptance_eligible": False,
    }


def issue(args: argparse.Namespace) -> dict[str, Any]:
    validated = gate.validate_gate(
        live_result=args.live_result,
        candidate_so=args.candidate_so,
        candidate_source=args.candidate_source,
        build_attestation=args.build_attestation,
        patch_source=args.patch_source,
        runner=args.qualification_runner,
        subset=args.subset,
        vocab_blocks=args.vocab_blocks,
        fa2_so=args.fa2_so,
        expected_source_commit=args.expected_source_commit,
        final_flush=args.final_flush,
        boundary_snapshot=args.boundary_snapshot,
        chat_traffic_audit=args.chat_traffic_audit,
        repo=args.repo,
    )
    payload = _credential_from_gate(validated)
    gate._write(args.out, payload)
    return payload


def validate_credential(
    *,
    credential_path: Path,
    expected_credential_sha256: str,
    candidate_so: Path,
    candidate_source: Path,
    build_attestation: Path,
    patch_source: Path,
    qualification_runner: Path,
    subset: Path,
    vocab_blocks: Path,
    fa2_so: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    expected_digest = gate._sha256(expected_credential_sha256, "expected credential")
    payload, raw = _canonical_load(credential_path, "DFWD U8 credential")
    if terminal.sha256_file(credential_path) != expected_digest:
        raise ValueError("DFWD U8 credential SHA-256 mismatch")
    _exact_dict(payload, CREDENTIAL_KEYS, "DFWD U8 credential")
    inputs = _exact_dict(payload.get("inputs"), INPUT_KEYS, "credential inputs")
    evidence = _exact_dict(
        payload.get("evidence_sha256"), EVIDENCE_KEYS, "credential evidence"
    )
    for key, value in evidence.items():
        gate._sha256(value, f"credential evidence {key}")
    expected_source_commit = gate._commit(
        expected_source_commit, "expected runtime source"
    )
    if (
        payload.get("schema") != CREDENTIAL_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("suite") != "SWE-Verified"
        or payload.get("qualification_schema") != gate.GATE_SCHEMA
        or payload.get("qualification_task_id") != gate.EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{gate.EXPECTED_INSTANCE}"
        or payload.get("source_commit") != expected_source_commit
        or payload.get("selector") != SELECTOR
        or payload.get("serve_policy") != "candidate_only_after_internal_attestation"
        or payload.get("topology") != gate.EXPECTED_TOPOLOGY
        or payload.get("geometry") != gate.EXPECTED_GEOMETRY
        or payload.get("candidate") != gate.EXPECTED_CANDIDATE
        or payload.get("captured_mtp_depths") != [1, 2, 3, 4]
        or payload.get("comparison_scope") != gate.COMPARISON_SCOPE
        or payload.get("raw_bf16_mismatches") != 0
        or payload.get("incumbent_served_during_qualification") is not True
        or payload.get("candidate_returned_during_qualification") is not False
        or payload.get("graph_contract") != GRAPH_CONTRACT
        or payload.get("production_default_enabled") is not False
        or payload.get("timing_eligible") is not True
        or payload.get("performance_claim") is not False
        or payload.get("floor_acceptance_eligible") is not False
    ):
        raise ValueError("DFWD U8 production credential provenance drifted")

    paths = {
        "candidate_so_sha256": candidate_so,
        "candidate_source_sha256": candidate_source,
        "build_attestation_sha256": build_attestation,
        "patch_source_sha256": patch_source,
        "runner_sha256": qualification_runner,
        "subset_sha256": subset,
        "vocab_blocks_sha256": vocab_blocks,
        "fa2_sha256": fa2_so,
    }
    for label, path in paths.items():
        _regular(path, label)
        if terminal.sha256_file(path) != inputs.get(label):
            raise ValueError(f"DFWD U8 credential input drifted: {label}")
    if (
        inputs.get("candidate_so_bytes") != gate.EXPECTED_SO_BYTES
        or candidate_so.stat().st_size != gate.EXPECTED_SO_BYTES
        or inputs.get("candidate_so_sha256") != gate.EXPECTED_SO_SHA256
        or inputs.get("candidate_source_sha256") != gate.EXPECTED_SOURCE_SHA256
        or inputs.get("build_attestation_sha256")
        != gate.EXPECTED_BUILD_ATTESTATION_SHA256
        or inputs.get("subset_sha256") != gate.EXPECTED_SUBSET_SHA256
        or inputs.get("vocab_blocks_sha256") != gate.EXPECTED_VOCAB_BLOCKS_SHA256
        or inputs.get("fa2_sha256") != gate.EXPECTED_FA2_SHA256
    ):
        raise ValueError("DFWD U8 credential pinned identity drifted")
    return {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "credential_sha256": expected_digest,
        "credential_bytes": len(raw),
        "source_commit": expected_source_commit,
        "selector": SELECTOR,
        "graph_contract": GRAPH_CONTRACT,
        "performance_claim": False,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    return validate_credential(
        credential_path=args.credential,
        expected_credential_sha256=args.expected_credential_sha256,
        candidate_so=args.candidate_so,
        candidate_source=args.candidate_source,
        build_attestation=args.build_attestation,
        patch_source=args.patch_source,
        qualification_runner=args.qualification_runner,
        subset=args.subset,
        vocab_blocks=args.vocab_blocks,
        fa2_so=args.fa2_so,
        expected_source_commit=args.expected_source_commit,
    )


def validate_engagement(
    *,
    engagement_path: Path,
    expected_credential_sha256: str,
    expected_source_commit: str,
) -> dict[str, Any]:
    payload, _ = _canonical_load(engagement_path, "DFWD U8 engagement")
    _exact_dict(payload, ENGAGEMENT_KEYS, "DFWD U8 engagement")
    expected_credential_sha256 = gate._sha256(
        expected_credential_sha256, "engagement credential"
    )
    expected_source_commit = gate._commit(expected_source_commit, "engagement source")
    if (
        payload.get("schema") != ENGAGEMENT_SCHEMA
        or payload.get("status") != "ENGAGED"
        or payload.get("source_commit") != expected_source_commit
        or payload.get("candidate_so_sha256") != gate.EXPECTED_SO_SHA256
        or payload.get("candidate_source_sha256") != gate.EXPECTED_SOURCE_SHA256
        or payload.get("production_credential_sha256") != expected_credential_sha256
        or payload.get("geometry") != gate.EXPECTED_GEOMETRY
        or payload.get("qualification_candidate") != gate.EXPECTED_CANDIDATE
        or payload.get("selector") != SELECTOR
        or payload.get("selected_root_calls") != 1
        or payload.get("captured_loop_calls") != 4
        or payload.get("fallback_calls") != 0
        or type(payload.get("drafter_graph_id")) is not int
        or payload["drafter_graph_id"] <= 0
        or payload.get("drafter_graph_signature") != GRAPH_SIGNATURE
        or type(payload.get("observed_measured_replays_at_least")) is not int
        or payload["observed_measured_replays_at_least"] < 1
        or payload.get("capture_origin") not in ("measured", "unmeasured")
        or payload.get("execution_basis") != "cudagraph_replay"
        or type(payload.get("forward_step_index")) is not int
        or payload["forward_step_index"] < 0
        or payload.get("runtime_mode") != "FULL"
        or payload.get("candidate_served") is not True
        or payload.get("incumbent_head_calls") != 0
        or payload.get("steady_state_synchronizations") != 0
        or payload.get("performance_claim") is not False
    ):
        raise ValueError("DFWD U8 production engagement drifted")
    return payload


def engagement(args: argparse.Namespace) -> dict[str, Any]:
    payload = validate_engagement(
        engagement_path=args.engagement,
        expected_credential_sha256=args.expected_credential_sha256,
        expected_source_commit=args.expected_source_commit,
    )
    return {
        "schema": ENGAGEMENT_SCHEMA,
        "status": "PASS",
        "selector": payload["selector"],
        "candidate_served": True,
        "captured_loop_calls": 4,
        "performance_claim": False,
    }


def _evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--live-result", type=Path, required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--build-attestation", type=Path, required=True)
    parser.add_argument("--patch-source", type=Path, required=True)
    parser.add_argument("--qualification-runner", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--vocab-blocks", type=Path, required=True)
    parser.add_argument("--fa2-so", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue_parser = subparsers.add_parser("issue")
    _evidence_arguments(issue_parser)
    _input_arguments(issue_parser)
    issue_parser.add_argument("--out", type=Path, required=True)
    issue_parser.set_defaults(handler=issue)
    verify_parser = subparsers.add_parser("verify")
    _input_arguments(verify_parser)
    verify_parser.add_argument("--credential", type=Path, required=True)
    verify_parser.add_argument("--expected-credential-sha256", required=True)
    verify_parser.set_defaults(handler=verify)
    engagement_parser = subparsers.add_parser("engagement")
    engagement_parser.add_argument("--engagement", type=Path, required=True)
    engagement_parser.add_argument("--expected-credential-sha256", required=True)
    engagement_parser.add_argument("--expected-source-commit", required=True)
    engagement_parser.set_defaults(handler=engagement)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
