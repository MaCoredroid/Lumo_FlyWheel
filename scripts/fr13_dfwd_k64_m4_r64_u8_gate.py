#!/usr/bin/env python3
"""Validate the real exact4 B4 candidate-served K64 M4 DFWD quality gate."""

from __future__ import annotations

import argparse
import hashlib
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
floor_gate = importlib.import_module("fr13_floor_gate")

LIVE_SCHEMA = "fr13.fixed32.dfwd_k64_m4_r64_u8_quality.v2"
GATE_SCHEMA = "fr13.fixed32.dfwd_k64_m4_r64_u8_real_b4_gate.v2"
TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
SO_SHA256 = "6cb24782495ff1c1457ebbf9cbcfcd6ca7b372378d3b435f80054688432a365f"
SO_BYTES = 134320
SOURCE_SHA256 = "a52361be1c9052a46509cc230ea320c4beb6d15f261327edc835d8da3ae00d9e"
BUILD_SHA256 = "b31ba7fb24fce81b0dceb97d77134f21107511e97538be15cb778c6ac4da5926"
SUBSET_SHA256 = "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
BLOCKS_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
FA2_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
FA2_BYTES = 299183936
TAW_SOURCE_CONTRACT_SHA256 = (
    "c8e32edf98453234bbf870c878d9f452930515b185061c6b9840282618ede9c3"
)
DEPTHS = ("root", "mtp_depth_1", "mtp_depth_2", "mtp_depth_3", "mtp_depth_4")
COMPARISON_SCOPE = (
    "all four rows and all 65536 logits at each of five fixed "
    "K64/root1 draft-head sites"
)
WORKER_ENV_KEYS = (
    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_AB",
    "FR13_DRAFT_HEAD_M4_R64_U8_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION",
    "FR13_DRAFT_HEAD_M4_R64_U8_SO",
    "FR13_DRAFT_HEAD_M4_R64_U8_SO_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_BUILD_ATTESTATION_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_PATCH_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_RUNNER_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SUBSET_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_VOCAB_BLOCKS_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_FA2_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_TAW_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_SOURCE_COMMIT",
    "FR13_DRAFT_HEAD_M4_R64_U8_TASK_IDS",
    "FR13_DRAFT_HEAD_M4_R64_U8_LIVE_JSON",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_DRAFT_HEAD_M4_R64_U8_INTERNAL_PRODUCTION_ATTESTED",
    "FR13_DRAFT_HEAD_M4_R64_U8_PRODUCTION_ENGAGEMENT_JSON",
    "FR13_DRAFT_VOCAB_BLOCKS",
    "FR13_DRAFT_VOCAB_K",
    "FR13_DRAFT_VOCAB_ROOT",
)
TOPOLOGY = {
    "mode": "hydra27_fixed32",
    "batch_size": 4,
    "physical_rows_per_request": 32,
    "total_physical_rows": 128,
    "logical_drafts_per_request": 27,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "execution_basis": "FULL_AND_PIECEWISE_graph_replay",
}
GEOMETRY = {
    "batch_size": 4,
    "calls_per_event": 5,
    "depths": ["root", 1, 2, 3, 4],
    "input_shape": [4, 5120],
    "input_stride": [5120, 1],
    "weight_shape": [65536, 5120],
    "weight_stride": [5120, 1],
    "output_shape": [4, 65536],
    "output_stride": [65536, 1],
    "dtype": "torch.bfloat16",
}
CANDIDATE = {
    "operation": "fr13_bf16_k64_head::gemvx_m4_shuffle_r64_u8_out",
    "device": "sm121",
    "grid": [1024, 1, 1],
    "block": [16, 64, 1],
    "rows_per_cta": 64,
    "batch_rows": 4,
    "weight_reuse": 4,
    "lane_products_per_row": 320,
    "unroll_steps": 8,
    "independent_accumulators": 4,
    "reduction_strides": [8, 4, 2, 1],
    "served_rows": 4,
    "shadow_only": False,
}
PROPOSAL_DISTRIBUTION = {
    "candidate_logits_consumed": True,
    "draft_probs": None,
    "proposal_token_selector": "argmax_topk",
    "q_mix_definition": "target_overlap_normalized_over_draft_token_ids",
    "rejection_sampler": "fr13_fixed32_deterministic_multidraft",
}
IDENTITY_KEYS = frozenset(
    {
        "build_attestation_sha256",
        "candidate_so_bytes",
        "candidate_so_sha256",
        "candidate_source_sha256",
        "fa2_sha256",
        "patch_source_sha256",
        "runner_sha256",
        "source_commit",
        "subset_sha256",
        "taw_source_sha256",
        "task_ids",
        "vocab_blocks_sha256",
    }
)
LIVE_KEYS = frozenset(
    {
        "batch_size",
        "boundary_snapshot_sha256",
        "candidate",
        "candidate_returned",
        "captured_mtp_depths",
        "compared_bytes",
        "compared_elements",
        "complete_work_census_events",
        "completed_events",
        "comparison_scope",
        "concurrency",
        "events_sha256",
        "finalized_by_fixed32_flush",
        "flush_action",
        "flush_generation",
        "flush_nonce",
        "full_logit_comparisons",
        "geometry",
        "identities",
        "per_depth_full_logit_comparisons",
        "per_depth_raw_bf16_mismatches",
        "per_depth_nonfinite_logits",
        "performance_measurement",
        "producer_pid",
        "raw_bf16_mismatches",
        "nonfinite_logits",
        "qualification_policy",
        "proposal_distribution",
        "taw_exact_acceptance",
        "reference_always_served",
        "root_forward_steps",
        "schema",
        "served_return",
        "source_commit",
        "status",
        "suite",
        "task_ids",
        "task_markers",
        "timing_eligible",
        "topology",
        "work_census_last_event_index",
        "worker_env_bridge",
    }
)


def _exact(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact(left, right) for left, right in zip(value, expected)
        )
    return value == expected


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in terminal.HEX for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in terminal.HEX for char in value)
    ):
        raise ValueError("source commit is not lowercase SHA-1")
    return value


def validate_live_result(
    payload: dict[str, Any], *, expected_source_commit: str
) -> dict[str, Any]:
    expected_source_commit = _commit(expected_source_commit)
    if frozenset(payload) != LIVE_KEYS:
        raise ValueError("DFWD M4 U8 live result key set drifted")
    identities = payload.get("identities")
    bridge = payload.get("worker_env_bridge")
    if not isinstance(identities, dict) or frozenset(identities) != IDENTITY_KEYS:
        raise ValueError("DFWD M4 U8 identity key set drifted")
    if (
        not isinstance(bridge, dict)
        or frozenset(bridge)
        != frozenset(
            {"schema", "sidecar", "sidecar_sha256", "payload_sha256", "hydrated_keys"}
        )
        or bridge.get("schema")
        != "fr13.fixed32.dfwd_k64_m4_r64_u8_worker_env_bridge.v1"
        or bridge.get("sidecar") != "/logs/fr13_draft_head_m4_r64_u8.worker_env.json"
        or bridge.get("hydrated_keys") != list(WORKER_ENV_KEYS)
    ):
        raise ValueError("DFWD M4 U8 worker env bridge drifted")
    _sha(bridge.get("sidecar_sha256"), "worker sidecar")
    _sha(bridge.get("payload_sha256"), "worker payload")
    for key in IDENTITY_KEYS:
        if key.endswith("sha256"):
            _sha(identities[key], f"identity {key}")
    if (
        payload.get("schema") != LIVE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("suite") != "SWE-Verified"
        or not _exact(payload.get("task_ids"), list(TASK_IDS))
        or not _exact(
            payload.get("task_markers"),
            ["swe_verified:" + task for task in TASK_IDS],
        )
        or not _exact(payload.get("concurrency"), 4)
        or not _exact(payload.get("batch_size"), 4)
        or payload.get("source_commit") != expected_source_commit
        or identities.get("source_commit") != expected_source_commit
        or not _exact(identities.get("task_ids"), list(TASK_IDS))
        or identities.get("candidate_so_sha256") != SO_SHA256
        or not _exact(identities.get("candidate_so_bytes"), SO_BYTES)
        or identities.get("candidate_source_sha256") != SOURCE_SHA256
        or identities.get("build_attestation_sha256") != BUILD_SHA256
        or identities.get("subset_sha256") != SUBSET_SHA256
        or identities.get("vocab_blocks_sha256") != BLOCKS_SHA256
        or identities.get("fa2_sha256") != FA2_SHA256
        or not _exact(payload.get("topology"), TOPOLOGY)
        or not _exact(payload.get("geometry"), GEOMETRY)
        or not _exact(payload.get("candidate"), CANDIDATE)
        or payload.get("comparison_scope") != COMPARISON_SCOPE
        or not _exact(payload.get("captured_mtp_depths"), [1, 2, 3, 4])
        or payload.get("qualification_policy")
        != "lossless_deterministic_proposal_taw_exact_v1"
        or not _exact(payload.get("proposal_distribution"), PROPOSAL_DISTRIBUTION)
        or payload.get("reference_always_served") is not False
        or payload.get("candidate_returned") is not True
        or payload.get("served_return") != "candidate BF16 logits"
        or payload.get("performance_measurement") is not False
        or payload.get("timing_eligible") is not False
        or payload.get("finalized_by_fixed32_flush") is not True
        or payload.get("flush_action") != "final"
    ):
        raise ValueError("DFWD M4 U8 live PASS provenance drifted")
    events = payload.get("completed_events")
    if type(events) is not int or events < 1:
        raise ValueError("DFWD M4 U8 completed event count is not positive")
    if (
        not _exact(payload.get("complete_work_census_events"), events)
        or not _exact(payload.get("work_census_last_event_index"), events - 1)
        or not _exact(payload.get("root_forward_steps"), list(range(events)))
        or not _exact(
            payload.get("per_depth_full_logit_comparisons"),
            {label: events for label in DEPTHS},
        )
        or not isinstance(payload.get("per_depth_raw_bf16_mismatches"), dict)
        or frozenset(payload["per_depth_raw_bf16_mismatches"])
        != frozenset(DEPTHS)
        or any(
            type(value) is not int or value < 0
            for value in payload["per_depth_raw_bf16_mismatches"].values()
        )
        or not _exact(
            payload.get("per_depth_nonfinite_logits"),
            {label: 0 for label in DEPTHS},
        )
        or not _exact(payload.get("full_logit_comparisons"), events * 5)
        or not _exact(payload.get("compared_elements"), events * 5 * 4 * 65536)
        or not _exact(payload.get("compared_bytes"), events * 5 * 4 * 65536 * 2)
        or not _exact(
            payload.get("raw_bf16_mismatches"),
            sum(payload["per_depth_raw_bf16_mismatches"].values()),
        )
        or not _exact(payload.get("nonfinite_logits"), 0)
        or type(payload.get("flush_generation")) is not int
        or payload["flush_generation"] < 1
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
    ):
        raise ValueError("DFWD M4 U8 five-site byte/event census drifted")
    for key in ("events_sha256", "flush_nonce", "boundary_snapshot_sha256"):
        _sha(payload.get(key), key)
    _validate_taw_exact(payload.get("taw_exact_acceptance"), payload)
    return payload


def _validate_taw_exact(value: Any, live: dict[str, Any]) -> dict[str, Any]:
    events = live.get("completed_events")
    expected_marker = "swe_verified:campaign4_" + SUBSET_SHA256
    binding = value.get("candidate_token_source") if isinstance(value, dict) else None
    identities = live.get("identities")
    expected_keys = {
        "accept_decision_mismatches", "batch_size", "candidate_token_source",
        "comparison_events", "completed_events", "draft_probs", "events_sha256",
        "mode", "probability_mismatches", "product_mismatches",
        "reference_returned", "schema", "source_contract_schema",
        "source_contract_sha256", "status", "target_authority", "task_marker",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("schema")
        != "fr13.fixed32.taw_candidate_acceptance_census.v1"
        or value.get("status") != "PASS"
        or value.get("mode") != "hydra27_fixed32"
        or not _exact(value.get("batch_size"), 4)
        or not _exact(value.get("completed_events"), events)
        or not _exact(value.get("comparison_events"), events)
        or value.get("events_sha256") != live.get("events_sha256")
        or value.get("task_marker") != expected_marker
        or not _exact(
            binding,
            {
                "operation": CANDIDATE["operation"],
                "candidate_so_sha256": identities.get("candidate_so_sha256"),
                "candidate_source_sha256": identities.get(
                    "candidate_source_sha256"
                ),
                "task_ids": list(TASK_IDS),
            },
        )
        or value.get("draft_probs") is not None
        or value.get("target_authority") is not True
        or value.get("source_contract_schema") != "fr13-fixed32-taw-all-parent-v7"
        or value.get("source_contract_sha256") != TAW_SOURCE_CONTRACT_SHA256
        or value.get("probability_mismatches") != 0
        or value.get("product_mismatches") != 0
        or value.get("accept_decision_mismatches") != 0
        or value.get("reference_returned") is not True
    ):
        raise ValueError("DFWD M4 U8 TAW exact-acceptance census drifted")
    return value


def _validate_inputs(live: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    for label, path in paths.items():
        terminal.require_regular_file(path, label)
    observed = {key: terminal.sha256_file(path) for key, path in paths.items()}
    if observed != {key: live["identities"][key] for key in paths}:
        raise ValueError("DFWD M4 U8 binary/source/runner closure drifted")
    if (
        paths["candidate_so_sha256"].stat().st_size != SO_BYTES
        or paths["fa2_sha256"].stat().st_size != FA2_BYTES
        or observed["candidate_so_sha256"] != SO_SHA256
        or observed["candidate_source_sha256"] != SOURCE_SHA256
        or observed["build_attestation_sha256"] != BUILD_SHA256
        or observed["subset_sha256"] != SUBSET_SHA256
        or observed["vocab_blocks_sha256"] != BLOCKS_SHA256
        or observed["fa2_sha256"] != FA2_SHA256
    ):
        raise ValueError("DFWD M4 U8 pinned input identity drifted")
    attestation, _ = terminal.load_json(paths["build_attestation_sha256"])
    if (
        attestation.get("schema")
        != "fr13.fixed32.dfwd_k64_m4_r64_u8_sm121a_canonical_build.v1"
        or attestation.get("status")
        != "REPRODUCIBLE_CANONICAL_LINKED_BUILD_UNQUALIFIED"
        or attestation.get("production_default_enabled") is not False
        or attestation.get("runtime_wired") is not False
        or attestation.get("real_task_correctness") is not False
        or attestation.get("binary", {}).get("sha256") != SO_SHA256
        or not _exact(attestation.get("binary", {}).get("bytes"), SO_BYTES)
        or attestation.get("source", {}).get("sha256") != SOURCE_SHA256
    ):
        raise ValueError("DFWD M4 U8 build attestation drifted")
    return {**observed, "candidate_so_bytes": SO_BYTES}


def _validate_b4_terminal(
    *, live: dict[str, Any], final_flush: Path, boundary_snapshot: Path
) -> dict[str, str | int]:
    final, _ = terminal.load_json(final_flush)
    boundary, boundary_raw = terminal.load_json(boundary_snapshot)
    ack = final.get("ack")
    metrics = boundary.get("metrics")
    fixed32 = metrics.get("fixed32") if isinstance(metrics, dict) else None
    sfwd = metrics.get("sfwd") if isinstance(metrics, dict) else None
    dfwd = metrics.get("dfwd") if isinstance(metrics, dict) else None
    cfwd = metrics.get("cfwd") if isinstance(metrics, dict) else None
    events = int(live["completed_events"])
    expected_counters = {
        "pure_decode_forward_steps": events,
        "complete_work_census_events": events,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": events - 1,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    expected_fixed32 = {
        "pure_decode_forward_steps": events,
        "complete_work_census_events": events,
        "complete_spec_rows": events * 4,
        "spec_drafts": events * 4,
        "spec_tokens": events * 4 * 31,
        "batch_histogram": {"1": 0, "2": 0, "3": 0, "4": events},
        "first_forward_step": 0,
        "last_forward_step": events - 1,
        "events_sha256": live["events_sha256"],
    }
    boundary_sha = hashlib.sha256(boundary_raw).hexdigest()
    if (
        final.get("schema") != "fr13-fixed32-flush-client-result-v1"
        or not isinstance(ack, dict)
        or ack.get("schema") != "fr13-fixed32-flush-ack-v1"
        or ack.get("mode") != "hydra27_fixed32"
        or ack.get("status") != "ok"
        or ack.get("action") != "final"
        or ack.get("generation") != live["flush_generation"]
        or ack.get("nonce") != live["flush_nonce"]
        or ack.get("producer_pid") != live["producer_pid"]
        or ack.get("counters") != expected_counters
        or boundary.get("schema") != "fr13-fixed32-boundary-snapshot-v4"
        or boundary.get("mode") != "hydra27_fixed32"
        or boundary.get("action") != "final"
        or boundary.get("generation") != live["flush_generation"]
        or boundary.get("nonce") != live["flush_nonce"]
        or boundary.get("producer_pid") != live["producer_pid"]
        or boundary.get("counters") != expected_counters
        or fixed32 != expected_fixed32
        or not isinstance(sfwd, dict)
        or sfwd.get("steps") != events
        or sfwd.get("drafts") != events * 4
        or not isinstance(dfwd, dict)
        or dfwd.get("spans") != events
        or not isinstance(cfwd, dict)
        or cfwd.get("spans") != events
        or boundary_sha != live["boundary_snapshot_sha256"]
    ):
        raise ValueError("DFWD M4 U8 B4 terminal flush evidence drifted")
    return {
        "boundary_snapshot_sha256": boundary_sha,
        "completed_events": events,
        "events_sha256": live["events_sha256"],
        "flush_generation": live["flush_generation"],
    }


def _validate_b4_traffic(
    *, audit_path: Path, subset_path: Path, repo: Path
) -> tuple[dict[str, Any], str]:
    subset = floor_gate.validate_fixed32_run_subset(
        subset_path.resolve(strict=True), b1_diagnostic=False
    )
    if subset.get("task_ids") != list(TASK_IDS):
        raise ValueError("DFWD M4 U8 subset is not the canonical exact four-task set")
    pinned = floor_gate.pinned_dataset_record_digests(str(repo.resolve(strict=True)))
    dataset_hashes = {task: pinned[task] for task in TASK_IDS}
    expected = floor_gate.build_fixed32_chat_traffic_audit(
        audit_path.parent.resolve(strict=True),
        mode="hydra27_fixed32",
        subset=subset,
        dataset_record_digests=dataset_hashes,
        concurrency=4,
    )
    audit, raw = terminal.load_json(audit_path)
    if audit != expected:
        raise ValueError(
            "DFWD M4 U8 authenticated traffic audit differs from raw evidence: "
            + floor_gate.first_json_difference(audit, expected)
        )
    return audit, hashlib.sha256(raw).hexdigest()


def validate_gate(args: argparse.Namespace) -> dict[str, Any]:
    live, _ = terminal.load_json(args.live_result)
    validate_live_result(live, expected_source_commit=args.expected_source_commit)
    paths = {
        "candidate_so_sha256": args.candidate_so,
        "candidate_source_sha256": args.candidate_source,
        "build_attestation_sha256": args.build_attestation,
        "patch_source_sha256": args.patch_source,
        "runner_sha256": args.runner,
        "subset_sha256": args.subset,
        "vocab_blocks_sha256": args.vocab_blocks,
        "fa2_sha256": args.fa2_so,
        "taw_source_sha256": args.taw_source,
    }
    inputs = _validate_inputs(live, paths)
    evidence = _validate_b4_terminal(
        live=live,
        final_flush=args.final_flush,
        boundary_snapshot=args.boundary_snapshot,
    )
    audit, traffic_sha = _validate_b4_traffic(
        audit_path=args.chat_traffic_audit,
        subset_path=args.subset,
        repo=args.repo,
    )
    tasks = audit.get("tasks")
    expected_eval = {"verdict": "resolved", "passed": True, "harness_exit_code": 0}
    if (
        not isinstance(tasks, dict)
        or tuple(tasks) != TASK_IDS
        or any(
            tasks[task].get("terminal", {}).get("eval") != expected_eval
            for task in TASK_IDS
        )
    ):
        raise ValueError("DFWD M4 U8 exact four-task SWE-Verified set did not resolve")
    events = int(live["completed_events"])
    return {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "source_commit": args.expected_source_commit,
        "task_ids": list(TASK_IDS),
        "all_tasks_resolved": True,
        "topology": TOPOLOGY,
        "geometry": GEOMETRY,
        "candidate": CANDIDATE,
        "inputs": inputs,
        "live_result_sha256": terminal.sha256_file(args.live_result),
        "completed_events": events,
        "captured_mtp_depths": [1, 2, 3, 4],
        "per_depth_full_logit_comparisons": {label: events for label in DEPTHS},
        "per_depth_raw_bf16_mismatches": live[
            "per_depth_raw_bf16_mismatches"
        ],
        "per_depth_nonfinite_logits": {label: 0 for label in DEPTHS},
        "compared_elements": events * 5 * 4 * 65536,
        "compared_bytes": events * 5 * 4 * 65536 * 2,
        "raw_bf16_mismatches": live["raw_bf16_mismatches"],
        "nonfinite_logits": 0,
        "qualification_policy": "lossless_deterministic_proposal_taw_exact_v1",
        "proposal_distribution": PROPOSAL_DISTRIBUTION,
        "taw_exact_acceptance": live["taw_exact_acceptance"],
        "reference_always_served": False,
        "candidate_returned": True,
        "events_sha256": evidence["events_sha256"],
        "final_flush_sha256": terminal.sha256_file(args.final_flush),
        "boundary_snapshot_sha256": evidence["boundary_snapshot_sha256"],
        "chat_traffic_audit_sha256": traffic_sha,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-result", type=Path, required=True)
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
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--chat-traffic-audit", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_gate(args)
    if args.out.exists() or args.out.is_symlink():
        raise ValueError(f"refusing to replace gate result: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(terminal.canonical_bytes(payload) + b"\n")
    os.replace(temporary, args.out)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
