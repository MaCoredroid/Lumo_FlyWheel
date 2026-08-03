#!/usr/bin/env python3
"""Reduce one real-task fixed32 ordered-GDN diagnostic into a scoped credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fr13_fixed32_contract as fixed32_contract  # noqa: E402
import fr13_fixed32_work_census as work_census  # noqa: E402
import fr13_floor_gate as floor_gate  # noqa: E402


SCHEMA = "fr13.fixed32.gdn_single_launch.real_task_credential.v2"
LIVE_SCHEMA = "fr13.fixed32.gdn_single_launch.live_pass.v1"
CANDIDATE = "fixed32_gdn_single_launch_tree_v2"
REFERENCE = "fixed32_gdn_two_launch_reference_v1"
DATASET = "princeton-nlp/SWE-bench_Verified"
B1_TASK_IDS = ("astropy__astropy-12907",)
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
MODE = {
    "tail6_fixed32": {
        "topology": "Tail23",
        "slug": "tail23",
        "logical_drafts": 23,
        "valid_mask": 0x7A9CE7FF,
    },
    "hydra27_fixed32": {
        "topology": "Hydra27",
        "slug": "hydra27",
        "logical_drafts": 27,
        "valid_mask": 0x7ABDFFFF,
    },
}
ENTRYPOINT = {
    ("hydra27_fixed32", 1): "scripts/fr13_run_b1_gdn_single_launch_live_gate.sh",
    ("tail6_fixed32", 4): (
        "scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh"
    ),
    ("hydra27_fixed32", 4): (
        "scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh"
    ),
}
COMMON_RUNNER = "scripts/fr13_run_gdn_single_launch_live_gate.sh"
KERNEL_SOURCE = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_SOURCE = "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
SERVER_LAUNCHER = "scripts/fr13_launch_forked_fa2_tree_server.sh"
BLOCK_MAP = "scripts/fr13_dvk_subset_blocks.json"
VALIDATOR_SOURCES = (
    "scripts/fr13_fixed32_contract.py",
    "scripts/fr13_fixed32_work_census.py",
    "scripts/fr13_floor_gate.py",
    "scripts/fr13_runtime_manifest.py",
)
BLOCK_MAP_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
TAW_ROUTE = work_census.TAW_ROUTE
HEX = frozenset("0123456789abcdef")


class GateError(RuntimeError):
    pass


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise GateError(f"{label} is missing: {path}") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
    ):
        raise GateError(f"{label} must be a singly-linked regular file: {path}")
    return path.read_bytes()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise GateError(f"{label} has duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GateError(f"{label} has non-finite value {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} JSON root is not an object")
    return payload, raw


def _canonical(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(raw)
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise GateError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_runtime_manifest(
    payload: dict[str, Any], *, required_closure: dict[str, str]
) -> str:
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
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
        or payload.get("sequence")
        != "scripts/fr13_fixed32_floor_timers_seq.sh"
        or payload.get("overall_canonical_sha256") != digest
    ):
        raise GateError("runtime manifest identity or canonical digest drifted")
    closures = payload.get("closures")
    if not isinstance(closures, dict):
        raise GateError("runtime manifest closures are missing")
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
            raise GateError(f"runtime manifest does not bind {path}")
    return digest


def _validate_health(payload: dict[str, Any], task_ids: list[str]) -> None:
    tasks = payload.get("tasks")
    if (
        payload.get("swe_orchestrator_rc") != 0
        or not isinstance(tasks, list)
        or [task.get("instance_id") for task in tasks] != task_ids
        or any(task.get("codex_timed_out") is not False for task in tasks)
        or any(task.get("verdict") == "missing" for task in tasks)
    ):
        raise GateError("health record does not prove all canonical tasks completed")


def _rebuild_traffic_audit(
    arm: Path,
    *,
    mode: str,
    subset: dict[str, Any],
    concurrency: int,
) -> tuple[dict[str, Any], bytes]:
    task_ids = list(subset["task_ids"])
    task_dirs = floor_gate.task_directories(
        arm,
        len(task_ids),
        expected_task_ids=task_ids,
    )
    dataset_hashes: dict[str, str] = {}
    for task_dir in task_dirs:
        metadata = floor_gate.exact_json(
            task_dir / "runner_metadata.json",
            label=f"{task_dir}:runner metadata",
        )
        digest = metadata.get("fixed32_dataset_record_sha256")
        dataset_hashes[task_dir.name] = _require_sha256(
            digest, f"{task_dir.name} dataset record"
        )
    expected = floor_gate.build_fixed32_chat_traffic_audit(
        arm,
        mode=mode,
        subset=subset,
        dataset_record_digests=dataset_hashes,
        concurrency=concurrency,
    )
    audit_path = arm / "fixed32_chat_traffic_audit.json"
    actual, raw = _load_json(audit_path, "authenticated traffic audit")
    if actual != expected:
        raise GateError(
            "authenticated traffic audit differs from raw task, Qwen, ingress, "
            "and census evidence: "
            + floor_gate.first_json_difference(actual, expected)
        )
    return actual, raw


def _validate_live_pass(
    payload: dict[str, Any],
    *,
    mode: str,
    batch: int,
    task_markers: frozenset[str],
    kernel_sha256: str,
    graph_signature: str,
) -> None:
    contract = MODE[mode]
    diagnostic_identity = (
        f"fixed32_gdn_single_launch_tree_v2:{contract['slug']}:b{batch}"
    )
    expected = {
        "schema": LIVE_SCHEMA,
        "status": "pass",
        "candidate": CANDIDATE,
        "source_sha256": kernel_sha256,
        "mode": mode,
        "batch_size": batch,
        "expected_batch": batch,
        "covered_batches": [batch],
        "records": 48,
        "physical_rows": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "compared_byte_surfaces": [
            "output",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "counter",
        ],
        "raw_byte_equal": True,
        "reference_served": True,
        "state_restored": True,
        "real_task_authenticated": True,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "logical_topology": contract["topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": contract["valid_mask"],
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "gate_mode": "post_first_measured_full_graph_replay",
        "diagnostic_identity": diagnostic_identity,
        "graph_signature": graph_signature,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise GateError(f"batch-specific GDN live PASS field drifted: {key}")
    if payload.get("task_marker") not in task_markers:
        raise GateError(
            "batch-specific GDN live PASS trigger is not a canonical task"
        )
    graph_id = payload.get("graph_id")
    if isinstance(graph_id, bool) or not isinstance(graph_id, int) or graph_id <= 0:
        raise GateError("batch-specific GDN live PASS graph_id is invalid")


def reduce(args: argparse.Namespace) -> dict[str, Any]:
    mode = args.mode
    batch = args.batch
    if (mode, batch) not in ENTRYPOINT:
        raise GateError("unsupported mode/batch credential scope")
    expected_tasks = list(B1_TASK_IDS if batch == 1 else EXACT4_TASK_IDS)
    subset = floor_gate.validate_fixed32_run_subset(
        args.subset,
        b1_diagnostic=batch == 1,
    )
    if subset["task_ids"] != expected_tasks:
        raise GateError("subset is not the exact expected task set")
    concurrency = batch
    arm = args.arm.resolve(strict=True)

    runtime_launch, runtime_launch_raw = _load_json(
        args.runtime_launch, "runtime launch manifest"
    )
    runtime_end, runtime_end_raw = _load_json(
        args.runtime_end, "runtime end manifest"
    )
    if runtime_launch_raw != runtime_end_raw or runtime_launch != runtime_end:
        raise GateError("runtime/source manifest changed from launch to end")
    external_launch, external_launch_raw = _load_json(
        args.external_launch, "external launch manifest"
    )
    external_end, external_end_raw = _load_json(
        args.external_end, "external end manifest"
    )
    if external_launch_raw != external_end_raw or external_launch != external_end:
        raise GateError("external image/model/FA2 manifest changed from launch to end")
    try:
        fixed32_contract.validate_external_manifest(external_end)
    except fixed32_contract.ContractError as error:
        raise GateError(f"external manifest is invalid: {error}") from error

    entrypoint = ENTRYPOINT[(mode, batch)]
    closure_paths = (
        entrypoint,
        COMMON_RUNNER,
        "scripts/fr13_gdn_single_launch_gate.py",
        *VALIDATOR_SOURCES,
        KERNEL_SOURCE,
        PATCHER_SOURCE,
        SERVER_LAUNCHER,
        BLOCK_MAP,
    )
    required_closure = {
        path: _sha256(_regular(REPO / path, f"source closure {path}"))
        for path in closure_paths
    }
    if required_closure[BLOCK_MAP] != BLOCK_MAP_SHA256:
        raise GateError("K64 block map identity drifted")
    runtime_digest = _validate_runtime_manifest(
        runtime_end,
        required_closure=required_closure,
    )

    health, health_raw = _load_json(arm / "health.json", "health record")
    _validate_health(health, expected_tasks)
    audit, audit_raw = _rebuild_traffic_audit(
        arm,
        mode=mode,
        subset=subset,
        concurrency=concurrency,
    )
    if audit.get("complete_stream", {}).get("complete_work_census_events", 0) <= 0:
        raise GateError("authenticated campaign has no complete work-census events")

    census_path = arm / "logs" / "fr13_fixed32_work_census.jsonl"
    census_raw = _regular(census_path, "fixed32 work census")
    try:
        work_report = work_census.validate_arm(
            work_census.load_jsonl_bytes(census_raw, source=str(census_path)),
            expected_mode=mode,
            expected_route=TAW_ROUTE,
            required_batches=(batch,),
        )
    except work_census.CensusError as error:
        raise GateError(f"graph/work census is invalid: {error}") from error
    forward_rows = work_report["terminal_summary"]["forward_graph_registry"]
    forward_row = next(
        (row for row in forward_rows if row.get("batch_size") == batch),
        None,
    )
    if not isinstance(forward_row, dict):
        raise GateError(f"graph/work census has no B{batch} FULL graph")
    graph_signature = _require_sha256(
        forward_row.get("graph_signature"), "FULL graph signature"
    )

    live, live_raw = _load_json(args.live_pass, "GDN single-launch live PASS")
    task_markers = frozenset(
        f"swe_verified:{task_id}" for task_id in expected_tasks
    )
    kernel_sha256 = required_closure[KERNEL_SOURCE]
    _validate_live_pass(
        live,
        mode=mode,
        batch=batch,
        task_markers=task_markers,
        kernel_sha256=kernel_sha256,
        graph_signature=graph_signature,
    )

    contract = MODE[mode]
    credential = {
        "schema": SCHEMA,
        "status": "PASS",
        "credential_scope": f"{contract['slug']}:b{batch}",
        "diagnostic_identity": live["diagnostic_identity"],
        "run_classification": (
            "one_real_swe_verified_b1_graph_byte_diagnostic"
            if batch == 1
            else "real_swe_verified_exact4_b4_graph_byte_diagnostic"
        ),
        "mode": mode,
        "logical_topology": contract["topology"],
        "logical_drafts": contract["logical_drafts"],
        "valid_mask": contract["valid_mask"],
        "expected_batch": batch,
        "batch_size": batch,
        "concurrency": concurrency,
        "task_ids": expected_tasks,
        "trigger_task_marker": live["task_marker"],
        "subset_sha256": subset["sha256"],
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "physical_rows": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "reference_served": True,
        "state_restored": True,
        "raw_byte_equal": True,
        "graph_id": live["graph_id"],
        "graph_signature": graph_signature,
        "work_census_event_count": work_report["event_count"],
        "work_census_sha256": _sha256(census_raw),
        "normalized_work_signature_sha256": work_report[
            "normalized_work_signature_sha256"
        ],
        "task_completion_verified": True,
        "finalized_ingress_verified": True,
        "qwen_compaction_algebra_replayed": True,
        "qwen_per_task_binding_verified": True,
        "runtime_launch_end_stable": True,
        "external_launch_end_stable": True,
        "graph_work_census_verified": True,
        "batch_specific_pass_verified": True,
        "production_enabled": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "floor_acceptance_eligible": False,
        "source_commit": args.source_commit,
        "kernel_source_sha256": kernel_sha256,
        "runtime_manifest_canonical_sha256": runtime_digest,
        "runtime_manifest_sha256": _sha256(runtime_end_raw),
        "external_manifest_sha256": _sha256(external_end_raw),
        "entrypoint_sha256": required_closure[entrypoint],
        "common_runner_sha256": required_closure[COMMON_RUNNER],
        "reducer_sha256": required_closure[
            "scripts/fr13_gdn_single_launch_gate.py"
        ],
        "live_pass_sha256": _sha256(live_raw),
        "health_sha256": _sha256(health_raw),
        "authenticated_traffic_audit_sha256": _sha256(audit_raw),
    }
    if re.fullmatch(r"[0-9a-f]{40}", args.source_commit) is None:
        raise GateError("source commit is not a full lowercase Git object ID")
    _atomic_write(args.output, _canonical(credential))
    return credential


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 4), required=True)
    parser.add_argument("--live-pass", type=Path, required=True)
    parser.add_argument("--runtime-launch", type=Path, required=True)
    parser.add_argument("--runtime-end", type=Path, required=True)
    parser.add_argument("--external-launch", type=Path, required=True)
    parser.add_argument("--external-end", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    credential = reduce(args)
    print(json.dumps(credential, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
