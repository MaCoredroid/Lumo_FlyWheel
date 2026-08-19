#!/usr/bin/env python3
"""Validate the real B1 candidate-served fixed-K64 DFWD R64-U8 quality gate."""

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


LIVE_SCHEMA = "fr13.fixed32.dfwd_k64_m1_r64_u8_quality.v2"
GATE_SCHEMA = "fr13.fixed32.dfwd_k64_m1_r64_u8_real_b1_gate.v2"
EXPECTED_INSTANCE = "astropy__astropy-12907"
EXPECTED_SO_SHA256 = (
    "8b27df4f3c6a5a0574261ee984159582a87615c3e6d83f2a267f4fa46a3e421e"
)
EXPECTED_SO_BYTES = 117904
EXPECTED_SOURCE_SHA256 = (
    "af0044edd84ff58d353a816f6887894d05a62b221e0efa5af933c2c59676b01b"
)
EXPECTED_BUILD_ATTESTATION_SHA256 = (
    "e7ec95d1fff3b665373ad7b3a14f7e3fad346cf77a5f2f992a90a689e5672c8f"
)
EXPECTED_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
EXPECTED_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
EXPECTED_FA2_SHA256 = (
    "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
)
EXPECTED_FA2_BYTES = 299183936
EXPECTED_TAW_SOURCE_CONTRACT_SHA256 = (
    "491874e3ebbc53b83ce28a8cae505025fde36e56564da049ab0d582eaa4e7d5c"
)
COMPARISON_SCOPE = (
    "all 65536 logits in the fixed K64/root1 draft head; "
    "not the full model vocabulary"
)
DEPTH_LABELS = (
    "root",
    "mtp_depth_1",
    "mtp_depth_2",
    "mtp_depth_3",
    "mtp_depth_4",
)
WORKER_ENV_KEYS = (
    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB",
    "FR13_DRAFT_HEAD_M1_R64_U8_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_QUALITY_GATE",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION",
    "FR13_DRAFT_HEAD_M1_R64_U8_SO",
    "FR13_DRAFT_HEAD_M1_R64_U8_SO_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_BUILD_ATTESTATION_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_PATCH_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_RUNNER_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SUBSET_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_VOCAB_BLOCKS_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_FA2_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_TAW_SOURCE_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT",
    "FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID",
    "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256",
    "FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED",
    "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_ENGAGEMENT_JSON",
    "FR13_DRAFT_VOCAB_BLOCKS",
    "FR13_DRAFT_VOCAB_K",
    "FR13_DRAFT_VOCAB_ROOT",
)
WORKER_ENV_BRIDGE_KEYS = frozenset(
    {
        "hydrated_keys",
        "payload_sha256",
        "schema",
        "sidecar",
        "sidecar_sha256",
    }
)
EXPECTED_TOPOLOGY = {
    "mode": "hydra27_fixed32",
    "batch_size": 1,
    "physical_rows": 32,
    "logical_drafts": 27,
    "draft_vocab_k": 65536,
    "draft_vocab_root": 1,
    "execution_basis": "FULL_AND_PIECEWISE_graph_replay",
}
EXPECTED_GEOMETRY = {
    "batch_size": 1,
    "calls_per_event": 5,
    "depths": ["root", 1, 2, 3, 4],
    "input_shape": [1, 5120],
    "input_stride": [5120, 1],
    "weight_shape": [65536, 5120],
    "weight_stride": [5120, 1],
    "output_shape": [1, 65536],
    "output_stride": [65536, 1],
    "dtype": "torch.bfloat16",
}
EXPECTED_CANDIDATE = {
    "operation": "fr13_bf16_k64_head::gemvx_m1_shuffle_r64_u8_out",
    "device": "sm121",
    "grid": [1024, 1, 1],
    "block": [16, 64, 1],
    "rows_per_cta": 64,
    "lane_products": 320,
    "unroll_steps": 8,
    "single_accumulator": True,
    "reduction_strides": [8, 4, 2, 1],
    "served_rows": 1,
    "shadow_only": False,
}
EXPECTED_PROPOSAL_DISTRIBUTION = {
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
        "instance_id",
        "patch_source_sha256",
        "runner_sha256",
        "source_commit",
        "subset_sha256",
        "taw_source_sha256",
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
        "finalized_by_fixed32_flush",
        "flush_action",
        "flush_generation",
        "flush_nonce",
        "full_logit_comparisons",
        "geometry",
        "identities",
        "instance_id",
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
        "task_marker",
        "timing_eligible",
        "topology",
        "work_census_last_event_index",
        "worker_env_bridge",
        "events_sha256",
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


def _json_exact(value: Any, expected: Any) -> bool:
    """Compare JSON values without bool/int/float coercion."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _json_exact(value[key], expected_value)
            for key, expected_value in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _json_exact(item, expected_item)
            for item, expected_item in zip(value, expected)
        )
    return value == expected


def validate_live_result(
    payload: dict[str, Any], *, expected_source_commit: str
) -> dict[str, Any]:
    expected_source_commit = _commit(expected_source_commit, "expected source")
    if frozenset(payload) != LIVE_KEYS:
        raise ValueError("DFWD U8 live result key set drifted")
    identities = payload.get("identities")
    if not isinstance(identities, dict) or frozenset(identities) != IDENTITY_KEYS:
        raise ValueError("DFWD U8 live identity key set drifted")
    worker_env_bridge = payload.get("worker_env_bridge")
    if (
        not isinstance(worker_env_bridge, dict)
        or frozenset(worker_env_bridge) != WORKER_ENV_BRIDGE_KEYS
        or worker_env_bridge.get("schema")
        != "fr13.fixed32.dfwd_k64_m1_r64_u8_worker_env_bridge.v1"
        or worker_env_bridge.get("sidecar")
        != "/logs/fr13_draft_head_m1_r64_u8.worker_env.json"
        or worker_env_bridge.get("hydrated_keys") != list(WORKER_ENV_KEYS)
    ):
        raise ValueError("DFWD U8 worker env bridge drifted")
    _sha256(worker_env_bridge.get("sidecar_sha256"), "worker sidecar")
    _sha256(worker_env_bridge.get("payload_sha256"), "worker payload")
    for key in IDENTITY_KEYS:
        if key.endswith("sha256"):
            _sha256(identities[key], f"identities.{key}")
    _commit(identities.get("source_commit"), "identity source")
    if (
        payload.get("schema") != LIVE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or not _json_exact(payload.get("concurrency"), 1)
        or not _json_exact(payload.get("batch_size"), 1)
        or payload.get("source_commit") != expected_source_commit
        or identities.get("source_commit") != expected_source_commit
        or identities.get("instance_id") != EXPECTED_INSTANCE
        or identities.get("candidate_so_sha256") != EXPECTED_SO_SHA256
        or not _json_exact(
            identities.get("candidate_so_bytes"), EXPECTED_SO_BYTES
        )
        or identities.get("candidate_source_sha256")
        != EXPECTED_SOURCE_SHA256
        or identities.get("build_attestation_sha256")
        != EXPECTED_BUILD_ATTESTATION_SHA256
        or identities.get("subset_sha256") != EXPECTED_SUBSET_SHA256
        or identities.get("vocab_blocks_sha256")
        != EXPECTED_VOCAB_BLOCKS_SHA256
        or identities.get("fa2_sha256") != EXPECTED_FA2_SHA256
        or not _json_exact(payload.get("topology"), EXPECTED_TOPOLOGY)
        or not _json_exact(payload.get("geometry"), EXPECTED_GEOMETRY)
        or not _json_exact(payload.get("candidate"), EXPECTED_CANDIDATE)
        or payload.get("comparison_scope") != COMPARISON_SCOPE
        or not _json_exact(payload.get("captured_mtp_depths"), [1, 2, 3, 4])
        or payload.get("qualification_policy")
        != "lossless_deterministic_proposal_v1"
        or not _json_exact(
            payload.get("proposal_distribution"),
            EXPECTED_PROPOSAL_DISTRIBUTION,
        )
        or payload.get("reference_always_served") is not False
        or payload.get("candidate_returned") is not True
        or payload.get("served_return") != "candidate BF16 logits"
        or payload.get("performance_measurement") is not False
        or payload.get("timing_eligible") is not False
        or payload.get("finalized_by_fixed32_flush") is not True
        or payload.get("flush_action") != "final"
    ):
        raise ValueError("DFWD U8 live PASS provenance drifted")

    events = payload.get("completed_events")
    if type(events) is not int or events < 1:
        raise ValueError("DFWD U8 live event count is not positive")
    comparisons = payload.get("per_depth_full_logit_comparisons")
    mismatches = payload.get("per_depth_raw_bf16_mismatches")
    nonfinite = payload.get("per_depth_nonfinite_logits")
    expected_comparisons = {label: events for label in DEPTH_LABELS}
    if (
        not _json_exact(payload.get("complete_work_census_events"), events)
        or not _json_exact(
            payload.get("work_census_last_event_index"), events - 1
        )
        or not _json_exact(payload.get("root_forward_steps"), list(range(events)))
        or not _json_exact(comparisons, expected_comparisons)
        or not isinstance(mismatches, dict)
        or frozenset(mismatches) != frozenset(DEPTH_LABELS)
        or any(type(value) is not int or value < 0 for value in mismatches.values())
        or not _json_exact(nonfinite, {label: 0 for label in DEPTH_LABELS})
        or not _json_exact(payload.get("full_logit_comparisons"), events * 5)
        or not _json_exact(
            payload.get("compared_elements"), events * 5 * 65536
        )
        or not _json_exact(
            payload.get("compared_bytes"), events * 5 * 65536 * 2
        )
        or not _json_exact(
            payload.get("raw_bf16_mismatches"), sum(mismatches.values())
        )
        or not _json_exact(payload.get("nonfinite_logits"), 0)
        or type(payload.get("flush_generation")) is not int
        or payload["flush_generation"] < 1
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
    ):
        raise ValueError("DFWD U8 per-depth quality/event census drifted")
    for key in ("events_sha256", "flush_nonce", "boundary_snapshot_sha256"):
        _sha256(payload.get(key), key)
    _validate_taw_exact_acceptance(payload.get("taw_exact_acceptance"), payload)
    return payload


def _validate_taw_exact_acceptance(
    payload: Any, live: dict[str, Any]
) -> dict[str, Any] | None:
    if payload is None:
        return None
    expected_keys = frozenset(
        {
            "accept_decision_mismatches",
            "batch_size",
            "candidate_token_source",
            "comparison_events",
            "completed_events",
            "draft_probs",
            "events_sha256",
            "mode",
            "probability_mismatches",
            "product_mismatches",
            "reference_returned",
            "schema",
            "source_contract_schema",
            "source_contract_sha256",
            "status",
            "target_authority",
            "task_marker",
        }
    )
    events = live.get("completed_events")
    binding = payload.get("candidate_token_source") if isinstance(payload, dict) else None
    identities = live.get("identities")
    if (
        not isinstance(payload, dict)
        or frozenset(payload) != expected_keys
        or payload.get("schema")
        != "fr13.fixed32.taw_candidate_acceptance_census.v1"
        or payload.get("status") != "PASS"
        or payload.get("mode") != "hydra27_fixed32"
        or not _json_exact(payload.get("batch_size"), 1)
        or not _json_exact(payload.get("completed_events"), events)
        or not _json_exact(payload.get("comparison_events"), events)
        or payload.get("events_sha256") != live.get("events_sha256")
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or not isinstance(binding, dict)
        or not _json_exact(
            binding,
            {
                "operation": EXPECTED_CANDIDATE["operation"],
                "candidate_so_sha256": identities.get("candidate_so_sha256"),
                "candidate_source_sha256": identities.get(
                    "candidate_source_sha256"
                ),
                "task_ids": [EXPECTED_INSTANCE],
            },
        )
        or payload.get("draft_probs") is not None
        or payload.get("target_authority") is not True
        or payload.get("source_contract_schema")
        != "fr13-fixed32-taw-all-parent-v7"
        or payload.get("source_contract_sha256")
        != EXPECTED_TAW_SOURCE_CONTRACT_SHA256
        or payload.get("probability_mismatches") != 0
        or payload.get("product_mismatches") != 0
        or payload.get("accept_decision_mismatches") != 0
        or payload.get("reference_returned") is not True
    ):
        raise ValueError("DFWD U8 TAW exact-acceptance census drifted")
    return payload


def _validate_inputs(
    *,
    live: dict[str, Any],
    candidate_so: Path,
    candidate_source: Path,
    build_attestation: Path,
    patch_source: Path,
    runner: Path,
    subset: Path,
    vocab_blocks: Path,
    fa2_so: Path,
    taw_source: Path,
) -> dict[str, str | int]:
    paths = {
        "candidate_so_sha256": candidate_so,
        "candidate_source_sha256": candidate_source,
        "build_attestation_sha256": build_attestation,
        "patch_source_sha256": patch_source,
        "runner_sha256": runner,
        "subset_sha256": subset,
        "vocab_blocks_sha256": vocab_blocks,
        "fa2_sha256": fa2_so,
        "taw_source_sha256": taw_source,
    }
    for label, path in paths.items():
        terminal.require_regular_file(path, label)
    expected = live["identities"]
    observed = {key: terminal.sha256_file(path) for key, path in paths.items()}
    if observed != {key: expected[key] for key in paths}:
        raise ValueError("DFWD U8 source/binary/runner binding drifted")
    if (
        candidate_so.stat().st_size != EXPECTED_SO_BYTES
        or fa2_so.stat().st_size != EXPECTED_FA2_BYTES
        or observed["candidate_so_sha256"] != EXPECTED_SO_SHA256
        or observed["candidate_source_sha256"] != EXPECTED_SOURCE_SHA256
        or observed["build_attestation_sha256"]
        != EXPECTED_BUILD_ATTESTATION_SHA256
        or observed["subset_sha256"] != EXPECTED_SUBSET_SHA256
        or observed["vocab_blocks_sha256"] != EXPECTED_VOCAB_BLOCKS_SHA256
        or observed["fa2_sha256"] != EXPECTED_FA2_SHA256
    ):
        raise ValueError("DFWD U8 pinned input identity drifted")
    attestation, _ = terminal.load_json(build_attestation)
    binary = attestation.get("binary")
    source = attestation.get("source")
    if (
        attestation.get("schema")
        != "fr13.fixed32.dfwd_k64_m1_r64_u8_sm121a_build.v1"
        or attestation.get("status") != "BUILT_UNQUALIFIED"
        or attestation.get("production_default_enabled") is not False
        or attestation.get("runtime_wired") is not False
        or attestation.get("real_task_correctness") is not False
        or not isinstance(binary, dict)
        or binary.get("sha256") != EXPECTED_SO_SHA256
        or not _json_exact(binary.get("bytes"), EXPECTED_SO_BYTES)
        or not isinstance(source, dict)
        or source.get("sha256") != EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("DFWD U8 build attestation drifted")
    return {**observed, "candidate_so_bytes": EXPECTED_SO_BYTES}


def validate_gate(
    *,
    live_result: Path,
    candidate_so: Path,
    candidate_source: Path,
    build_attestation: Path,
    patch_source: Path,
    runner: Path,
    subset: Path,
    vocab_blocks: Path,
    fa2_so: Path,
    taw_source: Path,
    expected_source_commit: str,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    repo: Path,
) -> dict[str, Any]:
    live, _ = terminal.load_json(live_result)
    validate_live_result(live, expected_source_commit=expected_source_commit)
    inputs = _validate_inputs(
        live=live,
        candidate_so=candidate_so,
        candidate_source=candidate_source,
        build_attestation=build_attestation,
        patch_source=patch_source,
        runner=runner,
        subset=subset,
        vocab_blocks=vocab_blocks,
        fa2_so=fa2_so,
        taw_source=taw_source,
    )
    terminal_evidence = terminal.validate_live_evidence(
        live_payload=live,
        final_flush_path=final_flush,
        boundary_snapshot_path=boundary_snapshot,
    )
    traffic = terminal.validate_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        expected_events=int(live["completed_events"]),
    )
    terminal.validate_rebuilt_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        repo=repo,
    )
    audit, _ = terminal.load_json(chat_traffic_audit)
    evaluation = audit["tasks"][EXPECTED_INSTANCE]["terminal"]["eval"]
    if evaluation != {
        "verdict": "resolved",
        "passed": True,
        "harness_exit_code": 0,
    }:
        raise ValueError("DFWD U8 real SWE-Verified task did not resolve")
    events = int(live["completed_events"])
    taw_exact_acceptance = _validate_taw_exact_acceptance(
        live["taw_exact_acceptance"], live
    )
    return {
        "schema": GATE_SCHEMA,
        "status": "PASS",
        "source_commit": expected_source_commit,
        "candidate": EXPECTED_CANDIDATE,
        "geometry": EXPECTED_GEOMETRY,
        "topology": EXPECTED_TOPOLOGY,
        "inputs": inputs,
        "live_result_sha256": terminal.sha256_file(live_result),
        "completed_events": events,
        "root_forward_steps": list(range(events)),
        "captured_mtp_depths": [1, 2, 3, 4],
        "comparison_scope": COMPARISON_SCOPE,
        "worker_env_bridge": live["worker_env_bridge"],
        "per_depth_full_logit_comparisons": {
            label: events for label in DEPTH_LABELS
        },
        "per_depth_raw_bf16_mismatches": live[
            "per_depth_raw_bf16_mismatches"
        ],
        "per_depth_nonfinite_logits": {label: 0 for label in DEPTH_LABELS},
        "compared_elements": events * 5 * 65536,
        "compared_bytes": events * 5 * 65536 * 2,
        "raw_bf16_mismatches": live["raw_bf16_mismatches"],
        "nonfinite_logits": 0,
        "qualification_policy": "lossless_deterministic_proposal_v1",
        "proposal_distribution": EXPECTED_PROPOSAL_DISTRIBUTION,
        "taw_exact_acceptance": taw_exact_acceptance,
        "reference_always_served": False,
        "candidate_returned": True,
        "task_resolved": True,
        "events_sha256": terminal_evidence["events_sha256"],
        "final_flush_sha256": terminal.sha256_file(final_flush),
        "boundary_snapshot_sha256": terminal_evidence[
            "boundary_snapshot_sha256"
        ],
        "chat_traffic_audit_sha256": traffic[
            "chat_traffic_audit_sha256"
        ],
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": taw_exact_acceptance is not None,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to replace DFWD U8 gate result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(terminal.canonical_bytes(payload) + b"\n")
    temporary.chmod(0o600)
    temporary.replace(path)


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
    payload = validate_gate(
        live_result=args.live_result,
        candidate_so=args.candidate_so,
        candidate_source=args.candidate_source,
        build_attestation=args.build_attestation,
        patch_source=args.patch_source,
        runner=args.runner,
        subset=args.subset,
        vocab_blocks=args.vocab_blocks,
        fa2_so=args.fa2_so,
        taw_source=args.taw_source,
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
