#!/usr/bin/env python3
"""Issue and verify the exact4 B4 K64/root1 M4 U8 credential."""

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

gate = importlib.import_module("fr13_dfwd_k64_m4_r64_u8_gate")
terminal = gate.terminal


CREDENTIAL_SCHEMA = "fr13.fixed32.dfwd_k64_m4_r64_u8_production_credential.v1"
VALIDATION_SCHEMA = "fr13.fixed32.dfwd_k64_m4_r64_u8_production_validation.v1"
SELECTOR = "fr13_bf16_k64_m4_r64_u8_direct"
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
        "taw_source_sha256",
        "vocab_blocks_sha256",
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
CREDENTIAL_KEYS = frozenset(
    {
        "all_tasks_resolved",
        "candidate",
        "candidate_returned_during_qualification",
        "captured_mtp_depths",
        "comparison_scope",
        "completed_events",
        "evidence_sha256",
        "floor_acceptance_eligible",
        "geometry",
        "graph_contract",
        "incumbent_served_during_qualification",
        "inputs",
        "nonfinite_logits",
        "performance_claim",
        "production_default_enabled",
        "proposal_distribution",
        "qualification_policy",
        "qualification_schema",
        "qualification_task_ids",
        "raw_bf16_equality_required",
        "raw_bf16_mismatches",
        "schema",
        "selector",
        "serve_policy",
        "source_commit",
        "status",
        "suite",
        "task_marker",
        "taw_exact_acceptance",
        "timing_eligible",
        "topology",
    }
)
GRAPH_CONTRACT = {
    "batch_size": 4,
    "calls_per_event": 5,
    "cudagraph_mode": "FULL_AND_PIECEWISE",
    "execution_basis": "cudagraph_replay",
    "mode": "hydra27_fixed32",
}


def _exact_dict(value: Any, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != keys:
        raise ValueError(f"{label} key set drifted")
    return value


def _canonical_load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload, raw = terminal.load_json(path)
    if raw != terminal.canonical_bytes(payload) + b"\n":
        raise ValueError(f"{label} is not canonical JSON")
    return payload, raw


def _credential_from_gate(validated: dict[str, Any]) -> dict[str, Any]:
    events = validated.get("completed_events")
    taw = validated.get("taw_exact_acceptance")
    if (
        validated.get("schema") != gate.GATE_SCHEMA
        or validated.get("status") != "PASS"
        or validated.get("task_ids") != list(gate.TASK_IDS)
        or validated.get("all_tasks_resolved") is not True
        or type(events) is not int
        or events < 1
        or validated.get("captured_mtp_depths") != [1, 2, 3, 4]
        or type(validated.get("raw_bf16_mismatches")) is not int
        or validated["raw_bf16_mismatches"] < 0
        or validated.get("nonfinite_logits") != 0
        or validated.get("qualification_policy")
        != "lossless_deterministic_proposal_taw_exact_v1"
        or not gate._exact(
            validated.get("proposal_distribution"), gate.PROPOSAL_DISTRIBUTION
        )
        or not isinstance(taw, dict)
        or taw.get("status") != "PASS"
        or taw.get("comparison_events") != events
        or taw.get("completed_events") != events
        or taw.get("probability_mismatches") != 0
        or taw.get("product_mismatches") != 0
        or taw.get("accept_decision_mismatches") != 0
        or taw.get("target_authority") is not True
        or validated.get("reference_always_served") is not False
        or validated.get("candidate_returned") is not True
        or validated.get("production_eligible") is not True
        or validated.get("performance_measurement") is not False
        or validated.get("timing_eligible") is not False
    ):
        raise ValueError("DFWD M4 U8 gate is not an exact candidate-served PASS")
    inputs = _exact_dict(validated.get("inputs"), INPUT_KEYS, "gate inputs")
    evidence = {
        "boundary_snapshot_sha256": validated["boundary_snapshot_sha256"],
        "chat_traffic_audit_sha256": validated["chat_traffic_audit_sha256"],
        "events_sha256": validated["events_sha256"],
        "final_flush_sha256": validated["final_flush_sha256"],
        "live_result_sha256": validated["live_result_sha256"],
    }
    for key, value in evidence.items():
        gate._sha(value, f"credential evidence {key}")
    return {
        "schema": CREDENTIAL_SCHEMA,
        "status": "PASS",
        "suite": "SWE-Verified",
        "qualification_schema": gate.GATE_SCHEMA,
        "qualification_task_ids": list(gate.TASK_IDS),
        "task_marker": "swe_verified:campaign4_" + gate.SUBSET_SHA256,
        "source_commit": validated["source_commit"],
        "selector": SELECTOR,
        "serve_policy": "candidate_only_after_internal_attestation",
        "topology": validated["topology"],
        "geometry": validated["geometry"],
        "candidate": validated["candidate"],
        "inputs": inputs,
        "evidence_sha256": evidence,
        "completed_events": events,
        "captured_mtp_depths": [1, 2, 3, 4],
        "comparison_scope": gate.COMPARISON_SCOPE,
        "raw_bf16_mismatches": validated["raw_bf16_mismatches"],
        "nonfinite_logits": 0,
        "qualification_policy": "lossless_deterministic_proposal_taw_exact_v1",
        "proposal_distribution": gate.PROPOSAL_DISTRIBUTION,
        "taw_exact_acceptance": taw,
        "raw_bf16_equality_required": False,
        "incumbent_served_during_qualification": False,
        "candidate_returned_during_qualification": True,
        "all_tasks_resolved": True,
        "graph_contract": GRAPH_CONTRACT,
        "production_default_enabled": False,
        "timing_eligible": True,
        "performance_claim": False,
        "floor_acceptance_eligible": False,
    }


def issue(args: argparse.Namespace) -> dict[str, Any]:
    validated = gate.validate_gate(args)
    payload = _credential_from_gate(validated)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"refusing to replace credential: {args.out}")
    _write(args.out, payload)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(terminal.canonical_bytes(payload) + b"\n")
    temporary.replace(path)


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
    taw_source: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    expected_digest = gate._sha(expected_credential_sha256, "expected credential")
    payload, raw = _canonical_load(credential_path, "DFWD M4 U8 credential")
    if terminal.sha256_file(credential_path) != expected_digest:
        raise ValueError("DFWD M4 U8 credential SHA-256 mismatch")
    _exact_dict(payload, CREDENTIAL_KEYS, "DFWD M4 U8 credential")
    inputs = _exact_dict(payload.get("inputs"), INPUT_KEYS, "credential inputs")
    evidence = _exact_dict(
        payload.get("evidence_sha256"), EVIDENCE_KEYS, "credential evidence"
    )
    for key, value in evidence.items():
        gate._sha(value, f"credential evidence {key}")
    expected_source_commit = gate._commit(expected_source_commit)
    taw = payload.get("taw_exact_acceptance")
    if (
        payload.get("schema") != CREDENTIAL_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("suite") != "SWE-Verified"
        or payload.get("qualification_schema") != gate.GATE_SCHEMA
        or payload.get("qualification_task_ids") != list(gate.TASK_IDS)
        or payload.get("task_marker")
        != "swe_verified:campaign4_" + gate.SUBSET_SHA256
        or payload.get("source_commit") != expected_source_commit
        or payload.get("selector") != SELECTOR
        or payload.get("serve_policy")
        != "candidate_only_after_internal_attestation"
        or not gate._exact(payload.get("topology"), gate.TOPOLOGY)
        or not gate._exact(payload.get("geometry"), gate.GEOMETRY)
        or not gate._exact(payload.get("candidate"), gate.CANDIDATE)
        or payload.get("captured_mtp_depths") != [1, 2, 3, 4]
        or payload.get("comparison_scope") != gate.COMPARISON_SCOPE
        or type(payload.get("completed_events")) is not int
        or payload["completed_events"] < 1
        or type(payload.get("raw_bf16_mismatches")) is not int
        or payload["raw_bf16_mismatches"] < 0
        or payload.get("nonfinite_logits") != 0
        or payload.get("qualification_policy")
        != "lossless_deterministic_proposal_taw_exact_v1"
        or not gate._exact(
            payload.get("proposal_distribution"), gate.PROPOSAL_DISTRIBUTION
        )
        or not isinstance(taw, dict)
        or taw.get("status") != "PASS"
        or taw.get("comparison_events") != payload["completed_events"]
        or taw.get("completed_events") != payload["completed_events"]
        or taw.get("probability_mismatches") != 0
        or taw.get("product_mismatches") != 0
        or taw.get("accept_decision_mismatches") != 0
        or taw.get("target_authority") is not True
        or payload.get("raw_bf16_equality_required") is not False
        or payload.get("incumbent_served_during_qualification") is not False
        or payload.get("candidate_returned_during_qualification") is not True
        or payload.get("all_tasks_resolved") is not True
        or not gate._exact(payload.get("graph_contract"), GRAPH_CONTRACT)
        or payload.get("production_default_enabled") is not False
        or payload.get("timing_eligible") is not True
        or payload.get("performance_claim") is not False
        or payload.get("floor_acceptance_eligible") is not False
    ):
        raise ValueError("DFWD M4 U8 production credential provenance drifted")
    paths = {
        "candidate_so_sha256": candidate_so,
        "candidate_source_sha256": candidate_source,
        "build_attestation_sha256": build_attestation,
        "patch_source_sha256": patch_source,
        "runner_sha256": qualification_runner,
        "subset_sha256": subset,
        "vocab_blocks_sha256": vocab_blocks,
        "fa2_sha256": fa2_so,
        "taw_source_sha256": taw_source,
    }
    for label, path in paths.items():
        terminal.require_regular_file(path, label)
        if terminal.sha256_file(path) != inputs.get(label):
            raise ValueError(f"DFWD M4 U8 credential input drifted: {label}")
    if (
        candidate_so.stat().st_size != gate.SO_BYTES
        or inputs.get("candidate_so_bytes") != gate.SO_BYTES
        or inputs.get("candidate_so_sha256") != gate.SO_SHA256
        or inputs.get("candidate_source_sha256") != gate.SOURCE_SHA256
        or inputs.get("build_attestation_sha256") != gate.BUILD_SHA256
        or inputs.get("subset_sha256") != gate.SUBSET_SHA256
        or inputs.get("vocab_blocks_sha256") != gate.BLOCKS_SHA256
        or inputs.get("fa2_sha256") != gate.FA2_SHA256
    ):
        raise ValueError("DFWD M4 U8 credential pinned identity drifted")
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
        qualification_runner=args.runner,
        subset=args.subset,
        vocab_blocks=args.vocab_blocks,
        fa2_so=args.fa2_so,
        taw_source=args.taw_source,
        expected_source_commit=args.expected_source_commit,
    )


def _input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--build-attestation", type=Path, required=True)
    parser.add_argument("--patch-source", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--vocab-blocks", type=Path, required=True)
    parser.add_argument("--fa2-so", type=Path, required=True)
    parser.add_argument("--taw-source", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue_parser = subparsers.add_parser("issue")
    issue_parser.add_argument("--live-result", type=Path, required=True)
    issue_parser.add_argument("--final-flush", type=Path, required=True)
    issue_parser.add_argument("--boundary-snapshot", type=Path, required=True)
    issue_parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    issue_parser.add_argument("--repo", type=Path, required=True)
    _input_arguments(issue_parser)
    issue_parser.add_argument("--out", type=Path, required=True)
    issue_parser.set_defaults(handler=issue)
    verify_parser = subparsers.add_parser("verify")
    _input_arguments(verify_parser)
    verify_parser.add_argument("--credential", type=Path, required=True)
    verify_parser.add_argument("--expected-credential-sha256", required=True)
    verify_parser.set_defaults(handler=verify)
    args = parser.parse_args()
    result = args.handler(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
