#!/usr/bin/env python3
"""Validate and bind a fixed32 GDN single-launch live qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from fr13_floor_gate import (  # noqa: E402
    build_fixed32_chat_traffic_audit,
    pinned_dataset_record_digests,
    validate_fixed32_pretask_zero_traffic,
    validate_fixed32_run_subset,
)
from fr13_fixed32_contract import (  # noqa: E402
    ContractError,
    validate_external_manifest,
)
from lumo_flywheel_serving.inference_proxy import (  # noqa: E402
    fixed32_task_key_id,
    verify_fixed32_ingress_ledger,
)


CANDIDATE = "fixed32_gdn_single_launch_tree_v2"
IDENTITY_SCHEMA = "fr13.fixed32.gdn_single_launch.identity.v2"
KERNEL = "_tree_gdn_kernel_fixed32_single_launch"
NODE_HELPER = "_tree_gdn_fixed32_single_launch_node"
CONTRACT_SHA256 = "ac748f003754a5f8562d864112c074450f376d10a9589d6047f1b88032f60393"
GROUPS_SHA256 = "cba9010f16772510ff6017e866a520552e7ada913bb786152133597cbc7c1f62"
EXECUTION_SHA256 = "80aed4d1a882ee4d4cde21dbf4314ed3abaae3f7553e35b6db5cd7574fe3b7db"
PARENT_SHA256 = "7abd25e38323d6c088eb627785b5c190b2e878b0a710bb349e2d690852a06ddd"
ANCESTRY_SHA256 = "90873d81e83ce1644ee4701e043b7e9d26e83b7a7ca752d538a0e6eed1946dad"
AUDITED_KERNEL_SOURCE_SHA256 = (
    "ca5ff6496c7cf3221996e6aa5971d36207e305e51f5c4a308f71d15165ab659a"
)
AUDITED_PATCHER_SOURCE_SHA256 = (
    "a32674b2b3dd8949c26001ca8a2664656373ab4890b207b6bddaca03367e6e94"
)
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
B1_SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
BLOCK_MAP_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
STOCK_FA2_SHA256 = "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d"
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
SURFACES = [
    "out",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
]
RESOURCE_AUDIT = (
    REPO / "results" / "fr13_fixed32_gdn_single_launch_tree_v2_sm121_audit_20260802"
)
RESOURCE_AUDIT_FILES = (
    "README.md",
    "compiler_audit.json",
    "manifest.json",
    "resources.tsv",
    "source_hashes.tsv",
    "verification.json",
)
CORE_RUNNER = REPO / "scripts" / "fr13_run_gdn_single_launch_live_gate.sh"


class VerdictError(RuntimeError):
    """A qualification input did not satisfy the fixed contract."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _ordered_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256_bytes(encoded)


def _regular_bytes(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> bytes:
    try:
        info = os.lstat(path)
    except OSError as error:
        raise VerdictError(f"required artifact is unavailable: {path}") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or info.st_size <= 0
        or info.st_size > max_bytes
    ):
        raise VerdictError(f"required artifact identity is invalid: {path}")
    payload = path.read_bytes()
    if len(payload) != info.st_size:
        raise VerdictError(f"required artifact changed while reading: {path}")
    return payload


def _validate_checksum_manifest(directory: Path) -> str:
    try:
        directory_info = os.lstat(directory)
    except OSError as error:
        raise VerdictError(f"audit directory is unavailable: {directory}") from error
    if not stat.S_ISDIR(directory_info.st_mode) or stat.S_ISLNK(directory_info.st_mode):
        raise VerdictError(f"audit directory identity is invalid: {directory}")

    checksum_raw = _regular_bytes(directory / "SHA256SUMS")
    try:
        checksum_text = checksum_raw.decode("ascii")
    except UnicodeError as error:
        raise VerdictError("resource audit checksum manifest is not ASCII") from error
    if not checksum_text.endswith("\n"):
        raise VerdictError("resource audit checksum manifest is not canonical")

    records: dict[str, str] = {}
    for line in checksum_text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise VerdictError("resource audit checksum row is malformed")
        digest, filename = line[:64], line[66:]
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not filename
            or Path(filename).name != filename
            or filename in records
        ):
            raise VerdictError("resource audit checksum row is unsafe")
        records[filename] = digest
    if tuple(sorted(records)) != tuple(sorted(RESOURCE_AUDIT_FILES)):
        raise VerdictError("resource audit checksum inventory is not canonical")
    for filename in RESOURCE_AUDIT_FILES:
        actual = _sha256_bytes(_regular_bytes(directory / filename))
        if actual != records[filename]:
            raise VerdictError(f"resource audit checksum failed: {filename}")
    return _sha256_bytes(checksum_raw)


def _json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular_bytes(path)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VerdictError(f"artifact is not ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise VerdictError(f"artifact root is not an object: {path}")
    return payload, raw


def single_launch_identity(
    *,
    batch_size: int,
    mode: str,
    source_sha256: str,
) -> dict[str, Any]:
    if batch_size not in (1, 4) or mode not in (
        "tail6_fixed32",
        "hydra27_fixed32",
    ):
        raise VerdictError("single-launch identity requires B1/B4 fixed32")
    identity: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "candidate": CANDIDATE,
        "kernel": KERNEL,
        "node_helper": NODE_HELPER,
        "mode": mode,
        "batch_size": batch_size,
        "physical_rows_per_request": 32,
        "block_v": 8,
        "physical_launches_per_layer": 1,
        "physical_programs_per_request_layer": 1,
        "physical_recurrence_critical_path": 32,
        "state_export_writes_per_request_layer": 0,
        "state_parent_reads_per_request_layer": 0,
        "authoritative_surfaces": SURFACES,
        "source_sha256": source_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "groups_sha256": GROUPS_SHA256,
        "execution_sha256": EXECUTION_SHA256,
        "parent_sha256": PARENT_SHA256,
        "ancestry_sha256": ANCESTRY_SHA256,
        "selector": "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=1",
    }
    identity["identity_sha256"] = _ordered_sha256(identity)
    return identity


def _runtime_manifest(
    path: Path,
    *,
    required: dict[str, str],
) -> tuple[str, bytes]:
    payload, raw = _json(path)
    claimed = payload.get("overall_canonical_sha256")
    unsigned = {
        key: value
        for key, value in payload.items()
        if key != "overall_canonical_sha256"
    }
    actual = _canonical_sha256(unsigned)
    if (
        payload.get("schema") != "fr13-runtime-manifest-v1"
        or payload.get("profile") != "fixed32"
        or payload.get("sequence") != "scripts/fr13_fixed32_floor_timers_seq.sh"
        or claimed != actual
    ):
        raise VerdictError("runtime manifest identity or digest is invalid")
    closures = payload.get("closures")
    if not isinstance(closures, dict):
        raise VerdictError("runtime manifest closures are missing")
    records = [
        record
        for group in closures.values()
        if isinstance(group, list)
        for record in group
        if isinstance(record, dict) and isinstance(record.get("path"), str)
    ]
    by_path = {record["path"]: record for record in records}
    for required_path, expected_sha256 in required.items():
        record = by_path.get(required_path)
        if not isinstance(record, dict) or record.get("sha256") != expected_sha256:
            raise VerdictError(f"runtime manifest does not bind {required_path}")
    return actual, raw


def _validate_resource_audit(source_sha256: str) -> str:
    checksum_sha256 = _validate_checksum_manifest(RESOURCE_AUDIT)
    verification, _raw = _json(RESOURCE_AUDIT / "verification.json")
    source_rows = _regular_bytes(RESOURCE_AUDIT / "source_hashes.tsv").decode("ascii")
    if (
        verification.get("status") != "PASS"
        or verification.get("checks", {}).get("gpu_kernel_not_executed") is not True
        or f"candidate_kernel\tsrc/lumo_flywheel_serving/fr10_gdn_tree_kernel.py\t{source_sha256}\n"
        not in source_rows
    ):
        raise VerdictError("offline sm_121a resource audit is not source-bound")
    return checksum_sha256


def build_verdict(args: argparse.Namespace) -> dict[str, Any]:
    batch = int(args.batch_size)
    mode = str(args.mode)
    expected_tasks = EXACT4_TASK_IDS if batch == 4 else EXACT4_TASK_IDS[:1]
    expected_subset_sha256 = EXACT4_SUBSET_SHA256 if batch == 4 else B1_SUBSET_SHA256
    if args.subset_sha256 != expected_subset_sha256:
        raise VerdictError("qualification subset digest is not canonical")
    if args.block_map_sha256 != BLOCK_MAP_SHA256:
        raise VerdictError("K64 block-map digest is not canonical")
    if args.stock_fa2_sha256 != STOCK_FA2_SHA256:
        raise VerdictError("stock FA2 digest is not canonical")

    arm = args.arm_dir.resolve()
    kernel_path = args.kernel_source.resolve()
    runner_path = args.runner.resolve()
    verifier_path = Path(__file__).resolve()
    subset_path = args.subset.resolve()
    block_map_path = args.block_map.resolve()
    patcher_path = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
    launcher_path = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
    ingress_path = REPO / "src" / "lumo_flywheel_serving" / "inference_proxy.py"
    core_runner_path = CORE_RUNNER.resolve()
    for path, expected in (
        (subset_path, args.subset_sha256),
        (block_map_path, args.block_map_sha256),
        (runner_path, args.runner_sha256),
    ):
        if _sha256_bytes(_regular_bytes(path)) != expected:
            raise VerdictError(f"qualification input changed: {path}")
    kernel_sha256 = _sha256_bytes(_regular_bytes(kernel_path))
    patcher_sha256 = _sha256_bytes(_regular_bytes(patcher_path))
    if kernel_sha256 != AUDITED_KERNEL_SOURCE_SHA256:
        raise VerdictError("kernel source differs from the sm_121a audited source")
    if patcher_sha256 != AUDITED_PATCHER_SOURCE_SHA256:
        raise VerdictError("runtime patcher differs from the sm_121a audited source")
    resource_audit_sha256 = _validate_resource_audit(kernel_sha256)

    subset = validate_fixed32_run_subset(
        subset_path,
        b1_diagnostic=batch == 1,
    )
    if (
        tuple(subset["task_ids"]) != expected_tasks
        or subset["sha256"] != args.subset_sha256
    ):
        raise VerdictError("parsed SWE-Verified task set is not canonical")
    validate_fixed32_pretask_zero_traffic(arm, mode=mode)
    expected_traffic = build_fixed32_chat_traffic_audit(
        arm,
        mode=mode,
        subset=subset,
        dataset_record_digests=pinned_dataset_record_digests(str(REPO)),
        concurrency=batch,
    )
    traffic, traffic_raw = _json(arm / "fixed32_chat_traffic_audit.json")
    if traffic != expected_traffic:
        raise VerdictError(
            "persisted real-task traffic audit differs from reconstruction"
        )

    health, _health_raw = _json(arm / "health.json")
    health_tasks = health.get("tasks")
    if (
        health.get("swe_orchestrator_rc") != 0
        or not isinstance(health_tasks, list)
        or any(not isinstance(record, dict) for record in health_tasks)
        or [record.get("instance_id") for record in health_tasks]
        != list(expected_tasks)
    ):
        raise VerdictError("SWE-Verified health record is incomplete")

    ledger_path = arm / "logs" / "fr13_fixed32_engine_ingress.jsonl"
    ledger_verification = verify_fixed32_ingress_ledger(
        ledger_path,
        expected_role="engine",
        require_finalized=True,
    )
    ledger_rows = [
        json.loads(line)
        for line in _regular_bytes(ledger_path).decode("ascii").splitlines()
    ]
    for task_id in expected_tasks:
        task_key = fixed32_task_key_id(task_id)
        accepted = sum(
            row.get("event") == "request_accepted"
            and row.get("task_key_id") == task_key
            for row in ledger_rows
        )
        completed = sum(
            row.get("event") == "request_complete"
            and row.get("task_key_id") == task_key
            and row.get("outcome") == "completed"
            for row in ledger_rows
        )
        if accepted <= 0 or completed != accepted:
            raise VerdictError(
                f"authenticated engine lifecycle is incomplete for {task_id}"
            )

    pass_name = f"fr13_fixed32_gdn_single_launch_b{batch}.live_pass.json"
    pass_path = arm / "logs" / pass_name
    live_pass, pass_raw = _json(pass_path)
    identity = single_launch_identity(
        batch_size=batch,
        mode=mode,
        source_sha256=kernel_sha256,
    )
    expected_pass: dict[str, Any] = {
        "schema": (
            "fr13.fixed32.gdn_single_launch.b1_live_pass.v2"
            if batch == 1
            else "fr13.fixed32.gdn_single_launch.b4_exact4_live_pass.v2"
        ),
        "status": "pass",
        "candidate": CANDIDATE,
        "identity_sha256": identity["identity_sha256"],
        "source_sha256": kernel_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "mode": mode,
        "batch_size": batch,
        "task_markers": [f"swe_verified:{task_id}" for task_id in expected_tasks],
        "layer_count": 48,
        "records_per_marker": 48,
        "reference_kernel_structure": "fixed32_path",
        "candidate_kernel_structure": KERNEL,
        "candidate_physical_launches_per_layer": 1,
        "candidate_state_export_writes": 0,
        "candidate_export_baseline_unchanged": True,
        "authoritative_surfaces": SURFACES,
        "raw_byte_equal": True,
        "reference_always_served_during_qualification": True,
        "state_restored": True,
    }
    if batch == 4:
        expected_pass["exact4_subset_sha256"] = EXACT4_SUBSET_SHA256
    if live_pass != expected_pass:
        raise VerdictError("single-launch live PASS differs from the source contract")

    runtime_path = args.runtime_manifest.resolve()
    external_path = args.external_manifest.resolve()
    relative_runner = runner_path.relative_to(REPO).as_posix()
    relative_verifier = verifier_path.relative_to(REPO).as_posix()
    required_closure = {
        relative_runner: args.runner_sha256,
        relative_verifier: _sha256_bytes(_regular_bytes(verifier_path)),
        core_runner_path.relative_to(REPO).as_posix(): _sha256_bytes(
            _regular_bytes(core_runner_path)
        ),
        kernel_path.relative_to(REPO).as_posix(): kernel_sha256,
        patcher_path.relative_to(REPO).as_posix(): patcher_sha256,
        launcher_path.relative_to(REPO).as_posix(): _sha256_bytes(
            _regular_bytes(launcher_path)
        ),
        ingress_path.relative_to(REPO).as_posix(): _sha256_bytes(
            _regular_bytes(ingress_path)
        ),
        subset_path.relative_to(REPO).as_posix(): args.subset_sha256,
        block_map_path.relative_to(REPO).as_posix(): args.block_map_sha256,
    }
    runtime_sha256, runtime_raw = _runtime_manifest(
        runtime_path,
        required=required_closure,
    )
    external, external_raw = _json(external_path)
    try:
        validate_external_manifest(external)
    except ContractError as error:
        raise VerdictError("external manifest is invalid") from error
    if external.get("forked_fa2", {}).get("sha256") != args.stock_fa2_sha256:
        raise VerdictError("external manifest does not bind the stock FA2")
    recorded_head = _regular_bytes(arm / "git_head.txt").decode("ascii").strip()
    if recorded_head != args.source_commit:
        raise VerdictError("run source commit differs from runner source commit")

    return {
        "schema": (
            "fr13.fixed32.gdn_single_launch.b1_k64_live_verdict.v1"
            if batch == 1
            else "fr13.fixed32.gdn_single_launch.b4_exact4_k64_live_verdict.v1"
        ),
        "status": "pass",
        "run_classification": (
            "one_real_swe_verified_k64_root1_b1_byte_diagnostic"
            if batch == 1
            else "real_swe_verified_exact4_k64_root1_b4_byte_diagnostic"
        ),
        "acceptance_valid": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_default_enabled": False,
        "candidate_shadow_only": True,
        "reference_always_served": True,
        "candidate_bytes_served_during_gate": False,
        "raw_prompt_response_published": False,
        "candidate": CANDIDATE,
        "mode": mode,
        "batch_size": batch,
        "physical_rows_per_request": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": args.block_map_sha256,
        "task_ids": list(expected_tasks),
        "task_markers": expected_pass["task_markers"],
        "subset_sha256": args.subset_sha256,
        "source_commit": args.source_commit,
        "kernel_source_sha256": kernel_sha256,
        "patcher_source_sha256": patcher_sha256,
        "contract_sha256": CONTRACT_SHA256,
        "identity_sha256": identity["identity_sha256"],
        "runner_sha256": args.runner_sha256,
        "verifier_sha256": required_closure[relative_verifier],
        "core_runner_sha256": required_closure[
            core_runner_path.relative_to(REPO).as_posix()
        ],
        "stock_fa2_sha256": args.stock_fa2_sha256,
        "live_pass_sha256": _sha256_bytes(pass_raw),
        "runtime_manifest_sha256": runtime_sha256,
        "runtime_manifest_file_sha256": _sha256_bytes(runtime_raw),
        "external_manifest_file_sha256": _sha256_bytes(external_raw),
        "traffic_audit_sha256": _sha256_bytes(traffic_raw),
        "engine_ledger_chain_head_sha256": ledger_verification["chain_head_sha256"],
        "sm121_resource_audit_sha256": resource_audit_sha256,
        "reference_physical_launches_per_layer": 2 * batch,
        "candidate_physical_launches_per_layer": 1,
        "authoritative_surfaces": SURFACES,
        "comparisons_per_marker": 48 * len(SURFACES),
        "raw_byte_equal": True,
        "candidate_export_baseline_unchanged": True,
        "state_restored": True,
        "no_positive_probe": True,
        "synthetic_traffic_admitted": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument(
        "--mode", choices=("tail6_fixed32", "hydra27_fixed32"), required=True
    )
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--subset-sha256", required=True)
    parser.add_argument("--block-map", type=Path, required=True)
    parser.add_argument("--block-map-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path, required=True)
    parser.add_argument("--stock-fa2-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        verdict = build_verdict(args)
    except (VerdictError, OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    encoded = (
        json.dumps(verdict, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, output)
    print(json.dumps(verdict, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
