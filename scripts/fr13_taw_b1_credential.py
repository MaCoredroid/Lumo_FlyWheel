#!/usr/bin/env python3
"""Issue and validate mode-bound source-v7 TAW B1 credentials."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


CREDENTIAL_SCHEMA = "fr13.fixed32.taw_source_v7.b1_credential.v1"
PAIR_SCHEMA = "fr13.fixed32.k64_physical32_fullstack.b1_pair.v1"
SOURCE_SCHEMA = "fr13-fixed32-taw-all-parent-v7"
SOURCE_CONTRACT_SHA256 = (
    "2b1cc55c6ec3d45c2d6ad0a21be4dc76685df4c974ae7fcfa421d5824a5c1ffb"
)
CANDIDATE = "fixed32_all_parent_commit_v2"
TASK_ID = "astropy__astropy-12907"
TASK_MARKER = f"swe_verified:{TASK_ID}"
EXACT4_TASK_IDS = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
STOCK_FA2_SHA256 = (
    "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
)
B1_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
BLOCK_MAP_CONTAINER = "/workspace/scripts/fr13_dvk_subset_blocks.json"
BLOCK_MAP_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
MANDATORY_WEIGHT_BYTES = 32_666_638_208
MANDATORY_WEIGHT_FLOOR_MS = 119.658015414
ONE_SIDED_U95_CAP_MS = 137.6067177261
MODE_CONTRACTS = {
    "tail6_fixed32": {
        "logical_topology": "Tail23",
        "logical_drafts": 23,
        "valid_mask": 0x7A9CE7FF,
    },
    "hydra27_fixed32": {
        "logical_topology": "Hydra27",
        "logical_drafts": 27,
        "valid_mask": 0x7ABDFFFF,
    },
}
B4_VERDICT_SCHEMAS = {
    "tail6_fixed32": "fr13.fixed32.tail23_all_parent.exact4_b4_live_gate.v1",
    "hydra27_fixed32": "fr13.fixed32.hydra27_all_parent.exact4_b4_live_gate.v1",
}
MERGE_BINDING_SCHEMA = "fr13.fixed32.taw_source_v7.b1_b4_merge_binding.v1"


class CredentialError(RuntimeError):
    """A source-v7 credential or timing input failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_regular(path: Path, *, label: str, limit: int = 16 << 20) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise CredentialError(f"{label} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    if not raw or len(raw) > limit:
        raise CredentialError(f"{label} is empty or exceeds {limit} bytes: {path}")
    return raw


def _load_json(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label=label)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"{label} is not canonical ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise CredentialError(f"{label} must be a JSON object")
    return payload, raw


def _atomic_write(path: Path, raw: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    os.chmod(temporary, mode)
    os.replace(temporary, path)


def _canonical_record_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256(raw)


def _load_source(path: Path):
    raw = _read_regular(path, label="TAW source", limit=8 << 20)
    spec = importlib.util.spec_from_file_location("fr13_taw_b1_credential_source", path)
    if spec is None or spec.loader is None:
        raise CredentialError("cannot import source-v7 TAW implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    topology = module._fr13_fixed32_topology()
    source_contract = module._fr13_fixed32_taw_source_contract(
        topology,
        batch_size=1,
    )
    if (
        module._FR13_FIXED32_TAW_SOURCE_SCHEMA != SOURCE_SCHEMA
        or module._FR13_FIXED32_TAW_SOURCE_SHA256 != SOURCE_CONTRACT_SHA256
        or source_contract.get("source_contract_sha256")
        != SOURCE_CONTRACT_SHA256
        or int(topology.PHYSICAL_DRAFTS) != 31
        or int(topology.PHYSICAL_ROWS) != 32
    ):
        raise CredentialError("source-v7 TAW implementation identity drifted")
    for mode, contract in MODE_CONTRACTS.items():
        if (
            int(topology.VALID_MASK_BY_MODE[mode]) != contract["valid_mask"]
            or module._fr13_fixed32_expected_active(topology, mode)
            != contract["logical_drafts"]
        ):
            raise CredentialError(f"source-v7 topology drifted for {mode}")
    return module, topology, _sha256(raw)


def _validate_live_bundle(
    payload: dict[str, Any],
    *,
    module,
    topology,
    mode: str,
    task_marker: str = TASK_MARKER,
) -> dict[str, Any]:
    contract = MODE_CONTRACTS[mode]
    batch_passes = payload.get("batch_passes")
    if (
        payload.get("schema")
        != "fr13.fixed32.taw_native_precompute.pass_bundle.v1"
        or payload.get("status") != "partial"
        or payload.get("candidate") != CANDIDATE
        or payload.get("source_contract_schema") != SOURCE_SCHEMA
        or payload.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256
        or payload.get("mode") != mode
        or payload.get("valid_mask") != contract["valid_mask"]
        or payload.get("required_production_batches") != [1, 4]
        or payload.get("qualified_batches") != [1]
        or not isinstance(batch_passes, dict)
        or set(batch_passes) != {"1"}
    ):
        raise CredentialError("live TAW B1 bundle is not a source-v7 partial pass")
    record = module._fr13_fixed32_taw_validate_pass_record(
        batch_passes["1"],
        topology,
        expected_mode=mode,
        expected_batch=1,
    )
    if (
        record.get("task_marker") != task_marker
        or record.get("evidence_route") != "full_graph_replay"
        or record.get("probability_mismatches") != 0
        or record.get("product_mismatches") != 0
        or record.get("reference_returned") is not True
        or record.get("candidate_returned") is not False
    ):
        raise CredentialError(
            "TAW B1 qualification was not an exact reference-served graph replay"
        )
    return record


def _validate_health(payload: dict[str, Any], *, task_ids: list[str]) -> None:
    tasks = payload.get("tasks")
    if (
        payload.get("swe_orchestrator_rc") != 0
        or not isinstance(tasks, list)
        or len(tasks) != len(task_ids)
        or sorted(task.get("instance_id") for task in tasks) != sorted(task_ids)
        or any(task.get("codex_timed_out") is not False for task in tasks)
        or any(task.get("verdict") == "missing" for task in tasks)
    ):
        raise CredentialError("health record does not prove clean terminal SWE tasks")


def _validate_traffic_audit(
    payload: dict[str, Any],
    *,
    mode: str,
    subset_sha256: str,
    task_ids: list[str],
) -> None:
    subset = payload.get("subset")
    checks = payload.get("checks")
    if (
        payload.get("schema") != "fr13-fixed32-chat-task-provenance-audit-v3"
        or payload.get("mode") != mode
        or not isinstance(subset, dict)
        or subset.get("sha256") != subset_sha256
        or subset.get("task_count") != len(task_ids)
        or sorted(subset.get("task_ids", [])) != sorted(task_ids)
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise CredentialError("authenticated traffic audit is incomplete")


def _validate_runtime_manifest(
    payload: dict[str, Any],
    *,
    required_closure: dict[str, str],
) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "overall_canonical_sha256"}
    digest = _sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    if (
        payload.get("schema") != "fr13-runtime-manifest-v1"
        or payload.get("profile") != "fixed32"
        or payload.get("overall_canonical_sha256") != digest
    ):
        raise CredentialError("runtime manifest identity or digest drifted")
    closures = payload.get("closures")
    if not isinstance(closures, dict):
        raise CredentialError("runtime manifest closures are missing")
    records = [
        record
        for group in closures.values()
        if isinstance(group, list)
        for record in group
        if isinstance(record, dict)
    ]
    by_path = {
        record.get("path"): record
        for record in records
        if isinstance(record.get("path"), str)
    }
    for path, expected in required_closure.items():
        if by_path.get(path, {}).get("sha256") != expected:
            raise CredentialError(f"runtime manifest does not bind {path}")
    return digest


def _mode_contract(mode: str) -> dict[str, Any]:
    try:
        return MODE_CONTRACTS[mode]
    except KeyError as error:
        raise CredentialError(f"unsupported fixed32 mode: {mode!r}") from error


def issue_credential(args: argparse.Namespace) -> dict[str, Any]:
    contract = _mode_contract(args.mode)
    source_path = Path(args.source)
    topology_path = Path(args.topology)
    subset_path = Path(args.subset)
    block_map_path = Path(args.block_map)
    runner_path = Path(args.runner)
    live_path = Path(args.live_bundle)
    runtime_path = Path(args.runtime_manifest)
    module, topology, source_file_sha256 = _load_source(source_path)

    subset, subset_raw = _load_json(subset_path, label="B1 task subset")
    task_ids = subset.get("instance_ids")
    if task_ids != [TASK_ID]:
        raise CredentialError("B1 gate is not bound to the canonical real SWE task")
    subset_sha256 = _sha256(subset_raw)
    if subset_sha256 != B1_SUBSET_SHA256:
        raise CredentialError("B1 task subset identity drifted")
    block_map_raw = _read_regular(block_map_path, label="K64 block map")
    if _sha256(block_map_raw) != BLOCK_MAP_SHA256:
        raise CredentialError("K64 block map identity drifted")

    live_payload, live_raw = _load_json(live_path, label="TAW B1 live bundle")
    record = _validate_live_bundle(
        live_payload,
        module=module,
        topology=topology,
        mode=args.mode,
    )
    health, health_raw = _load_json(Path(args.health), label="B1 health record")
    audit, audit_raw = _load_json(
        Path(args.traffic_audit),
        label="B1 authenticated traffic audit",
    )
    _validate_health(health, task_ids=task_ids)
    _validate_traffic_audit(
        audit,
        mode=args.mode,
        subset_sha256=subset_sha256,
        task_ids=task_ids,
    )

    runner_raw = _read_regular(runner_path, label="B1 gate runner")
    topology_raw = _read_regular(topology_path, label="fixed32 topology")
    helper_path = Path(__file__).resolve()
    helper_raw = _read_regular(helper_path, label="B1 credential helper")
    runtime, runtime_raw = _load_json(runtime_path, label="runtime manifest")
    required_closure = {
        "scripts/fr13_taw_b1_credential.py": _sha256(helper_raw),
        "scripts/fr13_run_b1_k64_taw_source_v7_gate.sh": _sha256(runner_raw),
        "scripts/fr13_device_multidraft_kernel.py": source_file_sha256,
        "scripts/fr13_fixed32_topology.py": _sha256(topology_raw),
        "scripts/fr13_dvk_subset_blocks.json": BLOCK_MAP_SHA256,
        "config/fr13_fixed32/subset_b1_diagnostic_one.json": subset_sha256,
    }
    runtime_digest = _validate_runtime_manifest(
        runtime,
        required_closure=required_closure,
    )

    curated_live_path = Path(args.curated_live_out)
    _atomic_write(curated_live_path, live_raw, mode=0o400)
    curated_raw = _read_regular(curated_live_path, label="curated TAW B1 bundle")
    if curated_raw != live_raw:
        raise CredentialError("curated B1 live bundle differs from validated input")

    credential = {
        "schema": CREDENTIAL_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_k64_b1_graph_byte_gate",
        "candidate": CANDIDATE,
        "source_contract_schema": SOURCE_SCHEMA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_file_sha256,
        "source_commit": args.source_commit,
        "mode": args.mode,
        "logical_topology": contract["logical_topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": task_ids,
        "task_marker": TASK_MARKER,
        "subset_sha256": subset_sha256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks": BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "mandatory_weight_bytes": MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": ONE_SIDED_U95_CAP_MS,
        "evidence_route": "full_graph_replay",
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "reference_returned": True,
        "candidate_returned": False,
        "production_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "live_bundle_sha256": _sha256(curated_raw),
        "live_b1_record_sha256": _canonical_record_sha256(record),
        "health_sha256": _sha256(health_raw),
        "authenticated_traffic_audit_sha256": _sha256(audit_raw),
        "runtime_manifest_sha256": _sha256(runtime_raw),
        "runtime_manifest_canonical_sha256": runtime_digest,
        "gate_runner_sha256": _sha256(runner_raw),
    }
    encoded = (
        json.dumps(credential, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(Path(args.out), encoded, mode=0o400)
    return credential


def validate_credential(
    path: Path,
    *,
    source_path: Path,
    mode: str,
) -> tuple[dict[str, Any], bytes, Any, Any]:
    contract = _mode_contract(mode)
    payload, raw = _load_json(path, label="TAW B1 credential")
    module, topology, source_file_sha256 = _load_source(source_path)
    expected = {
        "schema": CREDENTIAL_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_k64_b1_graph_byte_gate",
        "candidate": CANDIDATE,
        "source_contract_schema": SOURCE_SCHEMA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_file_sha256,
        "mode": mode,
        "logical_topology": contract["logical_topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": [TASK_ID],
        "task_marker": TASK_MARKER,
        "subset_sha256": B1_SUBSET_SHA256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks": BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "mandatory_weight_bytes": MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": ONE_SIDED_U95_CAP_MS,
        "evidence_route": "full_graph_replay",
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "reference_returned": True,
        "candidate_returned": False,
        "production_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    digest_keys = (
        "subset_sha256",
        "live_bundle_sha256",
        "live_b1_record_sha256",
        "health_sha256",
        "authenticated_traffic_audit_sha256",
        "runtime_manifest_sha256",
        "runtime_manifest_canonical_sha256",
        "gate_runner_sha256",
    )
    if any(
        not isinstance(payload.get(key), str)
        or len(payload[key]) != 64
        or any(character not in "0123456789abcdef" for character in payload[key])
        for key in digest_keys
    ):
        mismatches.append("digest_fields")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        mismatches.append("source_commit")
    if mismatches:
        raise CredentialError(
            "TAW B1 credential contract mismatch: " + ",".join(sorted(set(mismatches)))
        )
    return payload, raw, module, topology


def _validate_production_and_b1_binding(
    *,
    production_path: Path,
    credential: dict[str, Any],
    curated_live_path: Path,
    module,
    topology,
    mode: str,
) -> tuple[dict[str, Any], bytes]:
    live, live_raw = _load_json(curated_live_path, label="curated TAW B1 bundle")
    record = _validate_live_bundle(
        live,
        module=module,
        topology=topology,
        mode=mode,
    )
    if (
        _sha256(live_raw) != credential["live_bundle_sha256"]
        or _canonical_record_sha256(record)
        != credential["live_b1_record_sha256"]
    ):
        raise CredentialError("B1 credential does not bind the supplied live replay")
    production, production_raw = _load_json(
        production_path,
        label="TAW source-v7 production bundle",
    )
    module._fr13_fixed32_taw_native_production_pass(
        path=os.fspath(production_path),
        expected_mode=mode,
        expected_batch=1,
    )
    production_record = production.get("batch_passes", {}).get("1")
    if production_record != record:
        raise CredentialError(
            "production bundle B1 record is not the fresh mode-bound credential"
        )
    return production, production_raw


def _validate_reviewed_b4_gate(
    *,
    production_path: Path,
    verdict_path: Path,
    module,
    topology,
    source_file_sha256: str,
    mode: str,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes]:
    """Validate the hardened e0ac403c2 B4 pass-plus-verdict contract."""
    contract = MODE_CONTRACTS[mode]
    production, production_raw = _load_json(
        production_path,
        label="reviewed B4 TAW production bundle",
    )
    module._fr13_fixed32_taw_native_production_pass(
        path=os.fspath(production_path),
        expected_mode=mode,
        expected_batch=4,
    )
    if production.get("qualified_batches") != [1, 2, 3, 4]:
        raise CredentialError(
            "reviewed B4 TAW bundle lacks independent B1/B2/B3/B4 closure"
        )
    expected_marker = f"swe_verified:campaign4_{EXACT4_SUBSET_SHA256}"
    for batch in (1, 2, 3, 4):
        record = module._fr13_fixed32_taw_validate_pass_record(
            production["batch_passes"][str(batch)],
            topology,
            expected_mode=mode,
            expected_batch=batch,
        )
        if (
            record.get("task_marker") != expected_marker
            or record.get("evidence_route") != "full_graph_replay"
            or record.get("probability_mismatches") != 0
            or record.get("product_mismatches") != 0
            or record.get("reference_returned") is not True
            or record.get("candidate_returned") is not False
        ):
            raise CredentialError(f"reviewed B4 TAW B{batch} record is not exact")

    verdict, verdict_raw = _load_json(
        verdict_path,
        label="reviewed B4 TAW gate verdict",
    )
    production_sha256 = _sha256(production_raw)
    b4_source_commit = verdict.get("source_commit")
    digest_fields = (
        "campaign_proof_sha256",
        "runtime_manifest_sha256",
        "gate_runner_sha256",
    )
    expected = {
        "schema": B4_VERDICT_SCHEMAS[mode],
        "status": "pass",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "reference_always_served": True,
        "candidate_returned": False,
        "production_default_enabled": False,
        "raw_prompt_response_published": False,
        "candidate": CANDIDATE,
        "source_contract_schema": SOURCE_SCHEMA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_file_sha256,
        "mode": mode,
        "logical_topology": contract["logical_topology"],
        "active_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "qualified_batches": [1, 2, 3, 4],
        "required_production_batches": [1, 4],
        "independent_b1_record": True,
        "independent_b4_record": True,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks": BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "mandatory_weight_bytes": MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": ONE_SIDED_U95_CAP_MS,
        "subset_sha256": EXACT4_SUBSET_SHA256,
        "task_ids": EXACT4_TASK_IDS,
        "task_marker": expected_marker,
        "stock_fa2_sha256": STOCK_FA2_SHA256,
        "live_bundle_sha256": production_sha256,
        "production_bundle_sha256": production_sha256,
        "probability_mismatches": 0,
        "product_mismatches": 0,
    }
    mismatches = [key for key, value in expected.items() if verdict.get(key) != value]
    if (
        not isinstance(b4_source_commit, str)
        or len(b4_source_commit) != 40
        or any(character not in "0123456789abcdef" for character in b4_source_commit)
    ):
        mismatches.append("source_commit")
    if any(
        not isinstance(verdict.get(key), str)
        or len(verdict[key]) != 64
        or any(character not in "0123456789abcdef" for character in verdict[key])
        for key in digest_fields
    ):
        mismatches.append("verdict_digest_fields")
    if mismatches:
        raise CredentialError(
            "B4 TAW pass is not bound to the corrected exact4 verdict: "
            + ",".join(sorted(set(mismatches)))
        )
    return production, production_raw, verdict, verdict_raw


def _merge_binding_payload(
    *,
    mode: str,
    source_file_sha256: str,
    credential: dict[str, Any],
    credential_raw: bytes,
    b4_production: dict[str, Any],
    b4_production_raw: bytes,
    b4_verdict: dict[str, Any],
    b4_verdict_raw: bytes,
    merged: dict[str, Any],
    merged_raw: bytes,
) -> dict[str, Any]:
    contract = MODE_CONTRACTS[mode]
    return {
        "schema": MERGE_BINDING_SCHEMA,
        "status": "bound",
        "candidate": CANDIDATE,
        "source_contract_schema": SOURCE_SCHEMA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "source_file_sha256": source_file_sha256,
        "mode": mode,
        "logical_topology": contract["logical_topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "qualified_batches": merged["qualified_batches"],
        "required_production_batches": merged["required_production_batches"],
        "fresh_b1_credential_sha256": _sha256(credential_raw),
        "fresh_b1_live_bundle_sha256": credential["live_bundle_sha256"],
        "fresh_b1_record_sha256": credential["live_b1_record_sha256"],
        "reviewed_b4_production_bundle_sha256": _sha256(b4_production_raw),
        "reviewed_b4_gate_verdict_sha256": _sha256(b4_verdict_raw),
        "reviewed_b4_gate_verdict_schema": b4_verdict["schema"],
        "reviewed_b4_source_commit": b4_verdict["source_commit"],
        "reviewed_b4_record_sha256": _canonical_record_sha256(
            b4_production["batch_passes"]["4"]
        ),
        "preserved_reviewed_batches": [2, 3, 4],
        "merged_production_bundle_sha256": _sha256(merged_raw),
        "probability_mismatches": 0,
        "product_mismatches": 0,
        "reference_returned_in_all_gates": True,
        "production_default_enabled": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
    }


def merge_production(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source)
    credential, credential_raw, module, topology = validate_credential(
        Path(args.credential),
        source_path=source_path,
        mode=args.mode,
    )
    live, live_raw = _load_json(Path(args.b1_live_bundle), label="fresh B1 live bundle")
    b1_record = _validate_live_bundle(
        live,
        module=module,
        topology=topology,
        mode=args.mode,
    )
    if (
        _sha256(live_raw) != credential["live_bundle_sha256"]
        or _canonical_record_sha256(b1_record)
        != credential["live_b1_record_sha256"]
    ):
        raise CredentialError("fresh B1 bundle differs from its issued credential")

    b4, b4_raw, b4_verdict, b4_verdict_raw = _validate_reviewed_b4_gate(
        production_path=Path(args.b4_production_pass),
        verdict_path=Path(args.b4_gate_verdict),
        module=module,
        topology=topology,
        source_file_sha256=credential["source_file_sha256"],
        mode=args.mode,
    )
    merged = dict(b4)
    merged["batch_passes"] = dict(b4["batch_passes"])
    merged["batch_passes"]["1"] = b1_record
    encoded = (
        json.dumps(merged, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    output = Path(args.out)
    _atomic_write(output, encoded, mode=0o400)
    module._fr13_fixed32_taw_native_production_pass(
        path=os.fspath(output),
        expected_mode=args.mode,
        expected_batch=1,
    )
    _validate_production_and_b1_binding(
        production_path=output,
        credential=credential,
        curated_live_path=Path(args.b1_live_bundle),
        module=module,
        topology=topology,
        mode=args.mode,
    )
    binding = _merge_binding_payload(
        mode=args.mode,
        source_file_sha256=credential["source_file_sha256"],
        credential=credential,
        credential_raw=credential_raw,
        b4_production=b4,
        b4_production_raw=b4_raw,
        b4_verdict=b4_verdict,
        b4_verdict_raw=b4_verdict_raw,
        merged=merged,
        merged_raw=encoded,
    )
    binding_raw = (
        json.dumps(binding, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(Path(args.binding_out), binding_raw, mode=0o400)
    return {
        "schema": "fr13.fixed32.taw_source_v7.production_merge.v1",
        "status": "production_ready",
        "mode": args.mode,
        "production_bundle_sha256": _sha256(encoded),
        "merge_binding_sha256": _sha256(binding_raw),
        "reviewed_b4_gate_verdict_sha256": _sha256(b4_verdict_raw),
        "fresh_b1_record_sha256": credential["live_b1_record_sha256"],
        "qualified_batches": merged["qualified_batches"],
    }


def _validate_merge_chain(args: argparse.Namespace):
    credential, credential_raw, module, topology = validate_credential(
        Path(args.credential),
        source_path=Path(args.source),
        mode=args.mode,
    )
    production, production_raw = _validate_production_and_b1_binding(
        production_path=Path(args.production_pass),
        credential=credential,
        curated_live_path=Path(args.b1_live_bundle),
        module=module,
        topology=topology,
        mode=args.mode,
    )
    b4, b4_raw, b4_verdict, b4_verdict_raw = _validate_reviewed_b4_gate(
        production_path=Path(args.b4_production_pass),
        verdict_path=Path(args.b4_gate_verdict),
        module=module,
        topology=topology,
        source_file_sha256=credential["source_file_sha256"],
        mode=args.mode,
    )
    binding, binding_raw = _load_json(
        Path(args.merge_binding),
        label="TAW B1/B4 merge binding",
    )
    expected_binding = _merge_binding_payload(
        mode=args.mode,
        source_file_sha256=credential["source_file_sha256"],
        credential=credential,
        credential_raw=credential_raw,
        b4_production=b4,
        b4_production_raw=b4_raw,
        b4_verdict=b4_verdict,
        b4_verdict_raw=b4_verdict_raw,
        merged=production,
        merged_raw=production_raw,
    )
    if binding != expected_binding:
        raise CredentialError(
            "TAW merged production bundle is not bound to the reviewed B4 verdict"
        )
    return (
        credential,
        credential_raw,
        production,
        production_raw,
        b4_raw,
        b4_verdict_raw,
        binding_raw,
    )


def validate_production(args: argparse.Namespace) -> dict[str, Any]:
    (
        _,
        credential_raw,
        production,
        production_raw,
        b4_raw,
        b4_verdict_raw,
        binding_raw,
    ) = _validate_merge_chain(args)
    return {
        "schema": "fr13.fixed32.taw_source_v7.production_validation.v1",
        "status": "bound",
        "mode": args.mode,
        "credential_sha256": _sha256(credential_raw),
        "production_bundle_sha256": _sha256(production_raw),
        "reviewed_b4_production_bundle_sha256": _sha256(b4_raw),
        "reviewed_b4_gate_verdict_sha256": _sha256(b4_verdict_raw),
        "merge_binding_sha256": _sha256(binding_raw),
        "qualified_batches": production["qualified_batches"],
    }


def validate_reviewed_b4(args: argparse.Namespace) -> dict[str, Any]:
    module, topology, source_file_sha256 = _load_source(Path(args.source))
    production, production_raw, verdict, verdict_raw = _validate_reviewed_b4_gate(
        production_path=Path(args.production_pass),
        verdict_path=Path(args.gate_verdict),
        module=module,
        topology=topology,
        source_file_sha256=source_file_sha256,
        mode=args.mode,
    )
    return {
        "schema": "fr13.fixed32.taw_source_v7.reviewed_b4_validation.v1",
        "status": "bound",
        "mode": args.mode,
        "production_bundle_sha256": _sha256(production_raw),
        "gate_verdict_sha256": _sha256(verdict_raw),
        "gate_source_commit": verdict["source_commit"],
        "qualified_batches": production["qualified_batches"],
    }


def _positive(payload: dict[str, Any], key: str, *, label: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CredentialError(f"{label} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise CredentialError(f"{label} lacks positive finite {key}")
    return value


def _validate_measure(
    payload: dict[str, Any],
    *,
    label: str,
    task_ids: list[str],
    logical_drafts: int,
) -> dict[str, float]:
    if (
        payload.get("schema") != "fr13.measure.deploy_speed.v1"
        or payload.get("instrument") != "OFF"
        or payload.get("regime") != "deployment"
        or payload.get("batch_size") != 1
        or payload.get("n_tasks") != 4
        or sorted(payload.get("task_instance_ids", [])) != task_ids
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("mandatory_weight_bytes") != MANDATORY_WEIGHT_BYTES
    ):
        raise CredentialError(f"{label} measure is not exact4 K64 B1")
    wall_ms = _positive(payload, "step_wall_ms", label=label)
    wall_s_per_event = _positive(payload, "wall_s_per_event", label=label)
    events_per_step = _positive(payload, "events_per_step", label=label)
    full_wall_tps = _positive(payload, "measured_tps_fullstep_wall", label=label)
    accepted = _positive(payload, "accept_per_event", label=label)
    committed = _positive(payload, "committed_per_event", label=label)
    sfwd_s_per_event = _positive(payload, "s_per_fwd_gpu", label=label)
    sfwd_s_per_step = _positive(
        payload,
        "s_per_fwd_gpu_per_forward",
        label=label,
    )
    sfwd_ms = sfwd_s_per_step * 1000.0
    dfwd_ms = _positive(payload, "drafter_gpu_ms_per_step", label=label)
    cfwd_ms = _positive(payload, "committer_gpu_ms_per_step", label=label)
    if accepted > logical_drafts or not math.isclose(
        committed,
        accepted + 1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise CredentialError(f"{label} acceptance/commit accounting drifted")
    reconciled_tps = committed / wall_s_per_event
    if (
        not math.isclose(events_per_step, 1.0, rel_tol=0.0, abs_tol=1e-9)
        or not math.isclose(
            sfwd_s_per_step,
            sfwd_s_per_event * events_per_step,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            wall_ms,
            wall_s_per_event * events_per_step * 1000.0,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            full_wall_tps,
            reconciled_tps,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise CredentialError(f"{label} full-wall TPS does not reconcile")
    gpu_total = sfwd_ms + dfwd_ms + cfwd_ms
    other_wall = wall_ms - gpu_total
    if other_wall < 0 or not math.isfinite(other_wall):
        raise CredentialError(f"{label} phase timers exceed full-step wall")
    return {
        "full_step_wall_ms": wall_ms,
        "wall_ms_per_event": wall_s_per_event * 1000.0,
        "events_per_step": events_per_step,
        "full_wall_tps": full_wall_tps,
        "accepted_drafts_per_event": accepted,
        "committed_tokens_per_event": committed,
        "sfwd_ms": sfwd_ms,
        "sfwd_ms_per_event": sfwd_s_per_event * 1000.0,
        "dfwd_ms": dfwd_ms,
        "cfwd_ms": cfwd_ms,
        "gpu_component_total_ms": gpu_total,
        "other_wall_ms": other_wall,
        "wall_over_floor_ratio": wall_ms / MANDATORY_WEIGHT_FLOOR_MS,
        "wall_gap_to_cap_ms": wall_ms - ONE_SIDED_U95_CAP_MS,
    }


def _validate_engagements(
    *,
    sfwd: dict[str, Any],
    qrow: dict[str, Any],
    label: str,
) -> None:
    if (
        sfwd.get("schema")
        != "fr13.fixed32.sfwd_state_fusion.production_engagement.v1"
        or sfwd.get("candidate_served") is not True
        or sfwd.get("layer_count") != 48
        or sfwd.get("draft_vocab_root") != 1
        or sfwd.get("draft_vocab_k") != 65536
    ):
        raise CredentialError(f"{label} SFWD production engagement is incomplete")
    if (
        qrow.get("schema")
        != "fr13.fixed32.fa2_qrow16_eager_production_engagement.v1"
        or qrow.get("status") != "ENGAGED"
        or qrow.get("runtime_mode") != "EAGER"
        or qrow.get("layer_count") != 16
        or qrow.get("sfwd_state_fusion_production") is not True
    ):
        raise CredentialError(f"{label} qrow16 production engagement is incomplete")


def _load_work_census_module():
    path = Path(__file__).resolve().with_name("fr13_fixed32_work_census.py")
    scripts_dir = os.fspath(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "fr13_taw_b1_credential_work_census",
        path,
    )
    if spec is None or spec.loader is None:
        raise CredentialError("cannot import fixed32 work-census validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_taw_census(
    path: Path,
    *,
    expected_route: str,
    expected_mode: str,
) -> tuple[bytes, int]:
    raw = _read_regular(path, label="TAW production work census", limit=64 << 20)
    try:
        module = _load_work_census_module()
        records = module.load_jsonl(path)
        event_records, (terminal_raw, terminal_source) = module._split_terminal(
            records,
            expected_mode=expected_mode,
        )
        validated_events = []
        raw_events = []
        for event_raw, source in event_records:
            event = module.validate_event(event_raw, source=source)
            if (
                event.mode != expected_mode
                or event.batch_size != 1
                or event_raw.get("taw", {}).get("route") != expected_route
            ):
                raise CredentialError(
                    f"TAW work census event did not use {expected_route} at B1"
                )
            validated_events.append(event)
            raw_events.append(event_raw)
        if not validated_events:
            raise CredentialError("TAW work census contains no measured events")
        terminal = module._validate_terminal(
            terminal_raw,
            source=terminal_source,
            expected_mode=expected_mode,
            raw_events=raw_events,
            events=validated_events,
        )
    except CredentialError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RuntimeError) as error:
        raise CredentialError(f"TAW work census is invalid: {error}") from error
    if not raw:
        raise CredentialError(
            f"TAW work census did not use {expected_route} on every measured event"
        )
    return raw, int(terminal["event_count"])


def reduce_pair(args: argparse.Namespace) -> dict[str, Any]:
    contract = _mode_contract(args.mode)
    subset, subset_raw = _load_json(Path(args.subset), label="exact4 subset")
    task_ids = sorted(subset.get("instance_ids", []))
    if (
        task_ids != sorted(EXACT4_TASK_IDS)
        or _sha256(subset_raw) != EXACT4_SUBSET_SHA256
    ):
        raise CredentialError("timing pair is not the canonical exact4 task set")
    subset_sha256 = _sha256(subset_raw)

    (
        _,
        credential_raw,
        _,
        production_raw,
        b4_production_raw,
        b4_verdict_raw,
        merge_binding_raw,
    ) = _validate_merge_chain(args)

    stock_measure, stock_measure_raw = _load_json(
        Path(args.stock_measure), label="stock measure"
    )
    candidate_measure, candidate_measure_raw = _load_json(
        Path(args.candidate_measure), label="candidate measure"
    )
    stock_metrics = _validate_measure(
        stock_measure,
        label="stock",
        task_ids=task_ids,
        logical_drafts=contract["logical_drafts"],
    )
    candidate_metrics = _validate_measure(
        candidate_measure,
        label="candidate",
        task_ids=task_ids,
        logical_drafts=contract["logical_drafts"],
    )

    identity_hashes: dict[str, str] = {
        "credential_sha256": _sha256(credential_raw),
        "production_bundle_sha256": _sha256(production_raw),
        "reviewed_b4_production_bundle_sha256": _sha256(b4_production_raw),
        "reviewed_b4_gate_verdict_sha256": _sha256(b4_verdict_raw),
        "merge_binding_sha256": _sha256(merge_binding_raw),
        "stock_measure_sha256": _sha256(stock_measure_raw),
        "candidate_measure_sha256": _sha256(candidate_measure_raw),
    }
    stock_census_raw, stock_census_events = _validate_taw_census(
        Path(args.stock_taw_census),
        expected_route="fixed32_pytorch_exact_float_triton_integer_commit",
        expected_mode=args.mode,
    )
    candidate_census_raw, candidate_census_events = _validate_taw_census(
        Path(args.candidate_taw_census),
        expected_route="fixed32_native_precompute_production_candidate_return",
        expected_mode=args.mode,
    )
    identity_hashes["stock_taw_census_sha256"] = _sha256(stock_census_raw)
    identity_hashes["candidate_taw_census_sha256"] = _sha256(candidate_census_raw)
    census_by_arm = {
        "stock": (stock_census_raw, stock_census_events),
        "candidate": (candidate_census_raw, candidate_census_events),
    }
    for arm, health_name, audit_name, sfwd_name, qrow_name in (
        (
            "stock",
            args.stock_health,
            args.stock_audit,
            args.stock_sfwd_engagement,
            args.stock_qrow_engagement,
        ),
        (
            "candidate",
            args.candidate_health,
            args.candidate_audit,
            args.candidate_sfwd_engagement,
            args.candidate_qrow_engagement,
        ),
    ):
        health, health_raw = _load_json(Path(health_name), label=f"{arm} health")
        audit, audit_raw = _load_json(Path(audit_name), label=f"{arm} traffic audit")
        sfwd, sfwd_raw = _load_json(
            Path(sfwd_name), label=f"{arm} SFWD engagement"
        )
        qrow, qrow_raw = _load_json(
            Path(qrow_name), label=f"{arm} qrow engagement"
        )
        _validate_health(health, task_ids=task_ids)
        _validate_traffic_audit(
            audit,
            mode=args.mode,
            subset_sha256=subset_sha256,
            task_ids=task_ids,
        )
        census_raw, census_events = census_by_arm[arm]
        ingress = audit.get("ingress")
        complete_stream = audit.get("complete_stream")
        audit_census = ingress.get("census") if isinstance(ingress, dict) else None
        if (
            not isinstance(audit_census, dict)
            or not isinstance(complete_stream, dict)
            or audit_census.get("sha256") != _sha256(census_raw)
            or audit_census.get("bytes") != len(census_raw)
            or audit_census.get("event_count") != census_events
            or complete_stream.get("complete_work_census_events")
            != census_events
        ):
            raise CredentialError(
                f"{arm} authenticated traffic audit does not bind its work census"
            )
        _validate_engagements(sfwd=sfwd, qrow=qrow, label=arm)
        identity_hashes[f"{arm}_health_sha256"] = _sha256(health_raw)
        identity_hashes[f"{arm}_traffic_audit_sha256"] = _sha256(audit_raw)
        identity_hashes[f"{arm}_sfwd_engagement_sha256"] = _sha256(sfwd_raw)
        identity_hashes[f"{arm}_qrow_engagement_sha256"] = _sha256(qrow_raw)

    summary = {
        "schema": PAIR_SCHEMA,
        "status": "complete",
        "run_classification": "real_swe_verified_exact4_k64_b1_fullstack_pair",
        "task_ids": task_ids,
        "task_count": 4,
        "batch_size": 1,
        "concurrency": 1,
        "mode": args.mode,
        "logical_topology": contract["logical_topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": hex(contract["valid_mask"]),
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks": BLOCK_MAP_CONTAINER,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "source_commit": args.source_commit,
        "runner_sha256": args.runner_sha256,
        "subset_sha256": subset_sha256,
        "qrow16_production": True,
        "sfwd_state_fusion_production": True,
        "stock_all_parent_committer_production": False,
        "candidate_all_parent_committer_production": True,
        "only_arm_delta": "source_v7_all_parent_committer_production_0_to_1",
        "stock": stock_metrics,
        "candidate": candidate_metrics,
        "candidate_minus_stock": {
            "full_step_wall_ms": (
                candidate_metrics["full_step_wall_ms"]
                - stock_metrics["full_step_wall_ms"]
            ),
            "full_wall_tps": (
                candidate_metrics["full_wall_tps"] - stock_metrics["full_wall_tps"]
            ),
            "accepted_drafts_per_event": (
                candidate_metrics["accepted_drafts_per_event"]
                - stock_metrics["accepted_drafts_per_event"]
            ),
            "committed_tokens_per_event": (
                candidate_metrics["committed_tokens_per_event"]
                - stock_metrics["committed_tokens_per_event"]
            ),
            "sfwd_ms": candidate_metrics["sfwd_ms"] - stock_metrics["sfwd_ms"],
            "dfwd_ms": candidate_metrics["dfwd_ms"] - stock_metrics["dfwd_ms"],
            "cfwd_ms": candidate_metrics["cfwd_ms"] - stock_metrics["cfwd_ms"],
            "other_wall_ms": (
                candidate_metrics["other_wall_ms"] - stock_metrics["other_wall_ms"]
            ),
        },
        "candidate_to_stock_full_wall_tps_ratio": (
            candidate_metrics["full_wall_tps"] / stock_metrics["full_wall_tps"]
        ),
        "stock_to_candidate_wall_ratio": (
            stock_metrics["full_step_wall_ms"]
            / candidate_metrics["full_step_wall_ms"]
        ),
        "mandatory_weight_bytes": MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": ONE_SIDED_U95_CAP_MS,
        "timing_eligible": True,
        "formal_floor_acceptance_eligible": False,
        "formal_floor_acceptance_reason": (
            "paired exact4 screen only; canonical exact16 one-sided U95 remains required"
        ),
        "production_default_enabled": False,
        "identity_sha256": identity_hashes,
    }
    encoded = (
        json.dumps(summary, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    _atomic_write(Path(args.out), encoded)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    issue = subparsers.add_parser("issue")
    issue.add_argument("--mode", choices=sorted(MODE_CONTRACTS), required=True)
    issue.add_argument("--source", required=True)
    issue.add_argument("--topology", required=True)
    issue.add_argument("--subset", required=True)
    issue.add_argument("--block-map", required=True)
    issue.add_argument("--live-bundle", required=True)
    issue.add_argument("--runtime-manifest", required=True)
    issue.add_argument("--health", required=True)
    issue.add_argument("--traffic-audit", required=True)
    issue.add_argument("--runner", required=True)
    issue.add_argument("--source-commit", required=True)
    issue.add_argument("--curated-live-out", required=True)
    issue.add_argument("--out", required=True)

    validate = subparsers.add_parser("validate-credential")
    validate.add_argument("--mode", choices=sorted(MODE_CONTRACTS), required=True)
    validate.add_argument("--source", required=True)
    validate.add_argument("--credential", required=True)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--mode", choices=sorted(MODE_CONTRACTS), required=True)
    merge.add_argument("--source", required=True)
    merge.add_argument("--credential", required=True)
    merge.add_argument("--b1-live-bundle", required=True)
    merge.add_argument("--b4-production-pass", required=True)
    merge.add_argument("--out", required=True)

    merge.add_argument("--b4-gate-verdict", required=True)
    merge.add_argument("--binding-out", required=True)

    production = subparsers.add_parser("validate-production")
    production.add_argument("--mode", choices=sorted(MODE_CONTRACTS), required=True)
    production.add_argument("--source", required=True)
    production.add_argument("--credential", required=True)
    production.add_argument("--b1-live-bundle", required=True)
    production.add_argument("--b4-production-pass", required=True)
    production.add_argument("--b4-gate-verdict", required=True)
    production.add_argument("--merge-binding", required=True)
    production.add_argument("--production-pass", required=True)

    reviewed_b4 = subparsers.add_parser("validate-reviewed-b4")
    reviewed_b4.add_argument(
        "--mode", choices=sorted(MODE_CONTRACTS), required=True
    )
    reviewed_b4.add_argument("--source", required=True)
    reviewed_b4.add_argument("--production-pass", required=True)
    reviewed_b4.add_argument("--gate-verdict", required=True)

    reduce_command = subparsers.add_parser("reduce-pair")
    reduce_command.add_argument(
        "--mode", choices=sorted(MODE_CONTRACTS), required=True
    )
    reduce_command.add_argument("--source", required=True)
    reduce_command.add_argument("--subset", required=True)
    reduce_command.add_argument("--credential", required=True)
    reduce_command.add_argument("--b1-live-bundle", required=True)
    reduce_command.add_argument("--b4-production-pass", required=True)
    reduce_command.add_argument("--b4-gate-verdict", required=True)
    reduce_command.add_argument("--merge-binding", required=True)
    reduce_command.add_argument("--production-pass", required=True)
    reduce_command.add_argument("--stock-measure", required=True)
    reduce_command.add_argument("--candidate-measure", required=True)
    reduce_command.add_argument("--stock-health", required=True)
    reduce_command.add_argument("--candidate-health", required=True)
    reduce_command.add_argument("--stock-audit", required=True)
    reduce_command.add_argument("--candidate-audit", required=True)
    reduce_command.add_argument("--stock-sfwd-engagement", required=True)
    reduce_command.add_argument("--candidate-sfwd-engagement", required=True)
    reduce_command.add_argument("--stock-qrow-engagement", required=True)
    reduce_command.add_argument("--candidate-qrow-engagement", required=True)
    reduce_command.add_argument("--stock-taw-census", required=True)
    reduce_command.add_argument("--candidate-taw-census", required=True)
    reduce_command.add_argument("--source-commit", required=True)
    reduce_command.add_argument("--runner-sha256", required=True)
    reduce_command.add_argument("--out", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "issue":
        result = issue_credential(args)
    elif args.command == "validate-credential":
        payload, raw, _, _ = validate_credential(
            Path(args.credential),
            source_path=Path(args.source),
            mode=args.mode,
        )
        result = {
            "schema": "fr13.fixed32.taw_source_v7.b1_credential_validation.v1",
            "status": "bound",
            "mode": args.mode,
            "credential_sha256": _sha256(raw),
            "source_contract_sha256": payload["source_contract_sha256"],
        }
    elif args.command == "merge":
        result = merge_production(args)
    elif args.command == "validate-production":
        result = validate_production(args)
    elif args.command == "validate-reviewed-b4":
        result = validate_reviewed_b4(args)
    elif args.command == "reduce-pair":
        result = reduce_pair(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
