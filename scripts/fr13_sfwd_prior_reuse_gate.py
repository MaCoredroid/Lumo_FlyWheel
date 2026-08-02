#!/usr/bin/env python3
"""Create and validate source-bound SFWD prior-reuse gate artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_channel_serial_r32_c128_w4_u32x2_v1"
SOURCE_SCHEMA = "fr13.fixed32.sfwd_prior_reuse.source_manifest.v1"
REFERENCE_GDN_SOURCE_SHA256 = (
    "6c0f0ad607f15ea2727c2a9b244b1fe1c5ddb88268d70264c08f10470a5d2098"
)
SOURCE_FILES = (
    "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py",
    "src/lumo_flywheel_serving/fr13_sfwd_state_fusion_production.py",
    "src/lumo_flywheel_serving/inference_proxy.py",
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_run_b1_kernel_live_gate.sh",
    "scripts/fr13_run_b1_sfwd_prior_reuse_gate.sh",
    "scripts/fr13_sfwd_prior_reuse_gate.py",
    "scripts/fr13_dvk_subset_blocks.json",
    "scripts/run_swe_bench_q36_a.py",
)


class GateError(RuntimeError):
    pass


def _regular(path: Path, *, nonempty: bool = True) -> bytes:
    try:
        info = os.lstat(path)
        raw = path.read_bytes()
    except OSError as error:
        raise GateError(f"required artifact is unreadable: {path}: {error}") from error
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or info.st_nlink != 1
        or (nonempty and not raw)
    ):
        raise GateError(f"required artifact is not one regular file: {path}")
    return raw


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    )
    with tempfile.NamedTemporaryFile(
        "w", encoding="ascii", dir=path.parent, delete=False
    ) as handle:
        handle.write(raw)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_source_manifest(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    commit = str(args.source_commit)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise GateError("source commit must be a full lowercase Git SHA")
    try:
        head = (
            subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
            )
            .stdout.decode("ascii")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise GateError("cannot resolve the source repository HEAD") from error
    if head != commit:
        raise GateError("source commit does not equal repository HEAD")
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        raw = _regular(repo / relative)
        try:
            committed_raw = subprocess.run(
                ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise GateError(
                f"source-manifest file is not tracked at source commit: {relative}"
            ) from error
        if committed_raw != raw:
            raise GateError(f"working source differs from committed bytes: {relative}")
        files[relative] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    if (
        files["src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"]["sha256"]
        != REFERENCE_GDN_SOURCE_SHA256
    ):
        raise GateError("reference GDN source changed from the bound gate base")
    _write_json(
        Path(args.output).resolve(),
        {
            "schema": SOURCE_SCHEMA,
            "candidate": CANDIDATE,
            "source_commit": commit,
            "reference_gdn_source_bound": True,
            "files": files,
        },
    )


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"artifact is not ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise GateError(f"artifact is not a JSON object: {path}")
    return payload, raw


def validate_gate(args: argparse.Namespace) -> None:
    arm = Path(args.arm_dir).resolve()
    root = arm.parent
    logs = arm / "logs"
    source_commit = str(args.source_commit)
    task_id = str(args.task_id)
    marker = f"swe_verified:{task_id}"
    source_launch, source_launch_raw = _load_json(Path(args.manifest_launch))
    source_end, source_end_raw = _load_json(Path(args.manifest_end))
    if source_launch_raw != source_end_raw or source_launch != source_end:
        raise GateError("SFWD prior-reuse source manifest changed during the task")
    source_sha = hashlib.sha256(source_launch_raw).hexdigest()
    files = source_launch.get("files")
    if (
        source_launch.get("schema") != SOURCE_SCHEMA
        or source_launch.get("candidate") != CANDIDATE
        or source_launch.get("source_commit") != source_commit
        or source_launch.get("reference_gdn_source_bound") is not True
        or not isinstance(files, dict)
        or tuple(sorted(files)) != tuple(sorted(SOURCE_FILES))
        or files.get("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py", {}).get(
            "sha256"
        )
        != REFERENCE_GDN_SOURCE_SHA256
        or any(
            not isinstance(entry, dict)
            or not isinstance(entry.get("bytes"), int)
            or entry.get("bytes", 0) <= 0
            or not isinstance(entry.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            for entry in files.values()
        )
    ):
        raise GateError("SFWD prior-reuse source manifest contract mismatch")

    runtime_launch_raw = _regular(root / "runtime_manifest.at_launch.json")
    runtime_end_raw = _regular(root / "runtime_manifest.at_end.json")
    external_launch_raw = _regular(root / "external_manifest.at_launch.json")
    external_end_raw = _regular(root / "external_manifest.at_end.json")
    if runtime_launch_raw != runtime_end_raw:
        raise GateError("fixed32 runtime manifest changed during the task")
    if external_launch_raw != external_end_raw:
        raise GateError("fixed32 external manifest changed during the task")

    records_path = logs / "fr13_fixed32_sfwd_prior_reuse.byte_ab.jsonl"
    pass_path = logs / "fr13_fixed32_sfwd_prior_reuse.live_pass.json"
    marker_path = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
    records_raw = _regular(records_path)
    live_pass, pass_raw = _load_json(pass_path)
    marker_raw = _regular(marker_path)
    installed_manifest_path = (
        logs / "fr13_fixed32_sfwd_prior_reuse.source_manifest.json"
    )
    installed_manifest_raw = _regular(installed_manifest_path)
    installed_manifest_info = os.lstat(installed_manifest_path)
    if (
        installed_manifest_raw != source_launch_raw
        or stat.S_IMODE(installed_manifest_info.st_mode) != 0o400
    ):
        raise GateError("installed prior-reuse source manifest is not launch-bound")
    marker_info = os.lstat(marker_path)
    if stat.S_IMODE(marker_info.st_mode) != 0o444:
        raise GateError("authenticated real-event marker mode is not 0444")
    try:
        records = [
            json.loads(line) for line in records_raw.decode("ascii").splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError("prior-reuse comparison records are malformed") from error
    if not records or not all(isinstance(item, dict) for item in records):
        raise GateError("prior-reuse byte gate was vacuous")
    expected_record = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.byte_ab.v1",
        "status": "pass",
        "candidate": CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": source_sha,
        "candidate_source_sha256": files[
            "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
        ]["sha256"],
        "candidate_kernel_source_sha256": files[
            "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
        ]["sha256"],
        "task_marker": marker,
        "batch": 1,
        "physical_rows_per_request": 32,
        "conv_rows_per_program": 32,
        "conv_block_c": 128,
        "conv_num_warps": 4,
        "topology_host_validation": "exact_parent_each_launch",
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "x_shape": [32, 10240],
        "x_stride": [16384, 1],
        "out_stride": [10240, 1],
        "source_stage_shape": [36, 10240],
        "source_stage_stride": [10240, 1],
        "conv_weights_stride": [4, 1],
        "candidate_conv_launches_per_layer": 1,
        "gdn_level_path_programs": [1, 11],
        "gdn_physical_launches_per_layer": 2,
        "gdn_ring_export_unchanged": True,
        "gdn_flags_export_unchanged": True,
        "zero_diff": True,
        "real_task_authenticated": True,
        "reference_always_served": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
    }
    for record in records:
        for key, expected in expected_record.items():
            if record.get(key) != expected:
                raise GateError(f"comparison record {key} mismatch")
        comparisons = record.get("comparisons")
        if (
            not isinstance(comparisons, list)
            or [item.get("name") for item in comparisons]
            != ["conv_out", "commit_source_stage"]
            or any(item.get("byte_equal") is not True for item in comparisons)
        ):
            raise GateError("comparison surfaces are incomplete or unequal")
    record_layers = {record.get("layer_key") for record in records}
    record_pairs = {
        (record.get("layer_key"), record.get("layer_prefix_sha256"))
        for record in records
    }
    if (
        len(record_layers) != 48
        or len(record_pairs) != 48
        or any(
            not isinstance(key, str)
            or re.fullmatch(r"0x[0-9a-f]+", key) is None
            or not isinstance(prefix_sha, str)
            or re.fullmatch(r"[0-9a-f]{64}", prefix_sha) is None
            for key, prefix_sha in record_pairs
        )
    ):
        raise GateError("comparison records do not cover all 48 layers")

    expected_pass = {
        "schema": "fr13.fixed32.sfwd_prior_reuse.live_pass.v1",
        "status": "byte_pass_source_only",
        "run_classification": ("one_real_swe_verified_k64_root_b1_byte_diagnostic"),
        "candidate": CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": source_sha,
        "candidate_source_sha256": files[
            "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
        ]["sha256"],
        "candidate_kernel_source_sha256": files[
            "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
        ]["sha256"],
        "task_marker": marker,
        "batch": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "draft_vocab_blocks_sha256": (
            "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
        ),
        "layer_count": 48,
        "physical_rows_per_request": 32,
        "conv_rows_per_program": 32,
        "conv_block_c": 128,
        "conv_num_warps": 4,
        "topology_host_validation": "exact_parent_each_launch",
        "source_descriptor_device_validation": False,
        "source_descriptor_launcher_argument": False,
        "x_shape": [32, 10240],
        "x_stride": [16384, 1],
        "out_stride": [10240, 1],
        "source_stage_shape": [36, 10240],
        "source_stage_stride": [10240, 1],
        "conv_weights_stride": [4, 1],
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
    for key, expected in expected_pass.items():
        if live_pass.get(key) != expected:
            raise GateError(f"live PASS {key} mismatch")
    layers = live_pass.get("layers")
    if not isinstance(layers, list) or len(layers) != 48:
        raise GateError("live PASS does not bind 48 layers")
    pass_pairs = {
        (item.get("layer_key"), item.get("layer_prefix_sha256"))
        for item in layers
        if isinstance(item, dict)
    }
    if len(pass_pairs) != 48 or pass_pairs != record_pairs:
        raise GateError("record and live PASS layer identities differ")
    if marker_raw != (marker + "\n").encode("ascii"):
        raise GateError("authenticated real-event marker bytes mismatch")

    diagnostic, diagnostic_raw = _load_json(arm / "fixed32_b1_diagnostic.json")
    process_identity, process_raw = _load_json(arm / "fixed32_process_identity.json")
    terminal, terminal_raw = _load_json(arm / "fixed32_final_flush_skipped.json")
    traffic, traffic_raw = _load_json(arm / "fixed32_chat_traffic_audit_skipped.json")
    container_env_raw = _regular(arm / "container_env.txt")
    engine_raw = _regular(logs / "fr13_fixed32_engine_ingress.jsonl")
    docker_raw = _regular(arm / "docker_after_tasks.log")
    if diagnostic.get("task_ids") != [task_id]:
        raise GateError("B1 diagnostic task binding mismatch")
    if diagnostic.get("floor_acceptance_eligible") is not False:
        raise GateError("one-task diagnostic claimed floor acceptance")
    if terminal != {
        "schema": "fr13-fixed32-eager-kernel-terminal-v1",
        "run_classification": "eager_kernel_byte_diagnostic",
        "acceptance_valid": False,
        "flush_protocol_used": False,
    }:
        raise GateError("eager terminal no-flush marker mismatch")
    if (
        traffic.get("schema") != "fr13-fixed32-eager-kernel-traffic-audit-skip-v1"
        or traffic.get("run_classification") != "eager_kernel_byte_diagnostic"
        or traffic.get("acceptance_valid") is not False
        or traffic.get("authenticated_engine_ledger_snapshotted") is not True
        or traffic.get("graph_census_audit_used") is not False
    ):
        raise GateError("eager traffic skip marker mismatch")
    bracket_paths = list(
        arm.glob(f"swe_out/*/per_task/{task_id}/fixed32_task_boundary.json")
    )
    if len(bracket_paths) != 1:
        raise GateError("eager task bracket is missing or ambiguous")
    bracket, bracket_raw = _load_json(bracket_paths[0])
    if (
        bracket.get("schema") != "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1"
        or bracket.get("instance_id") != task_id
        or bracket.get("acceptance_valid") is not False
        or bracket.get("flush_protocol_used") is not False
    ):
        raise GateError("eager task bracket contract mismatch")

    container_env = container_env_raw.decode("ascii").splitlines()
    expected_env = (
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=1",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0",
        "FR13_FIXED32_SFWD_STATE_FUSION_TIMING_AB=0",
        f"FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_SHA256={source_sha}",
        f"FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_COMMIT={source_commit}",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "FR13_DRAFT_VOCAB_BLOCKS=/workspace/scripts/fr13_dvk_subset_blocks.json",
        "FR13_CONV_WB_BATCHED=1",
        "FR13_TREE_CONV_FUSED=1",
        "FR13_FIXED32_CUTLASS_WAVE=stock",
        "ENFORCE_EAGER=1",
    )
    if any(container_env.count(value) != 1 for value in expected_env):
        raise GateError("container environment did not preserve exclusive B1 gate")
    manifest_path_lines = [
        line
        for line in container_env
        if line.startswith("FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH=")
    ]
    if (
        len(manifest_path_lines) != 1
        or not manifest_path_lines[0].startswith(
            "FR13_FIXED32_SFWD_PRIOR_REUSE_SOURCE_MANIFEST_PATH=/workspace/output/"
        )
        or not manifest_path_lines[0].endswith(
            "/sfwd_prior_reuse_source_manifest.at_launch.json"
        )
        or "FR10_ALLOW_LINEAR_FALLBACK=1" in container_env
    ):
        raise GateError("source-manifest path or no-fallback environment drifted")
    pid1 = process_identity.get("pid1")
    argv = pid1.get("argv") if isinstance(pid1, dict) else None
    max_num_seqs_valid = False
    if isinstance(argv, list) and argv.count("--max-num-seqs") == 1:
        index = argv.index("--max-num-seqs")
        max_num_seqs_valid = index + 1 < len(argv) and argv[index + 1] == "1"
    if not max_num_seqs_valid:
        raise GateError("PID 1 did not use --max-num-seqs 1")
    docker_text = docker_raw.decode("utf-8", errors="replace")
    shim_lines = [
        line
        for line in docker_text.splitlines()
        if "[FR13_DRAFT_VOCAB] shim built K=65536 " in line
    ]
    root_lines = [
        line
        for line in docker_text.splitlines()
        if "[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 " in line
    ]
    if len(shim_lines) != 1 or "mode=gather" not in shim_lines[0]:
        raise GateError("K64 draft-vocabulary gather did not engage once")
    if len(root_lines) != 1 or "mode=gather" not in root_lines[0]:
        raise GateError("K64 root gather did not engage once")
    forbidden = (
        "[FR13_DRAFT_VOCAB] DISABLED",
        "FR10_ALLOW_LINEAR_FALLBACK=1",
        "linear fallback engaged",
    )
    if any(needle in docker_text for needle in forbidden):
        raise GateError("a forbidden full-vocabulary or linear fallback engaged")

    output = Path(args.output).resolve()
    _write_json(
        output,
        {
            "schema": "fr13.fixed32.sfwd_prior_reuse.k64_root_b1_gate.v1",
            "status": "pass",
            "run_classification": ("one_real_swe_verified_k64_root_b1_byte_diagnostic"),
            "candidate": CANDIDATE,
            "source_commit": source_commit,
            "source_manifest_sha256": source_sha,
            "reference_gdn_source_bound": True,
            "candidate_source_sha256": files[
                "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse.py"
            ]["sha256"],
            "candidate_kernel_source_sha256": files[
                "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py"
            ]["sha256"],
            "task_set": "one",
            "task_count": 1,
            "real_task_authenticated": True,
            "reference_returned": True,
            "no_fallback": True,
            "batch_size": 1,
            "physical_rows_per_request": 32,
            "conv_rows_per_program": 32,
            "conv_block_c": 128,
            "conv_num_warps": 4,
            "topology_host_validation": "exact_parent_each_launch",
            "source_descriptor_device_validation": False,
            "source_descriptor_launcher_argument": False,
            "x_shape": [32, 10240],
            "x_stride": [16384, 1],
            "out_stride": [10240, 1],
            "source_stage_shape": [36, 10240],
            "source_stage_stride": [10240, 1],
            "conv_weights_stride": [4, 1],
            "layer_count": 48,
            "comparison_count": len(records),
            "compared_byte_surfaces": ["conv_out", "commit_source_stage"],
            "timing_eligible": False,
            "floor_acceptance_eligible": False,
            "production_enabled": False,
            "records_sha256": hashlib.sha256(records_raw).hexdigest(),
            "live_pass_sha256": hashlib.sha256(pass_raw).hexdigest(),
            "runtime_manifest_sha256": hashlib.sha256(runtime_launch_raw).hexdigest(),
            "external_manifest_sha256": hashlib.sha256(external_launch_raw).hexdigest(),
            "diagnostic_sha256": hashlib.sha256(diagnostic_raw).hexdigest(),
            "task_bracket_sha256": hashlib.sha256(bracket_raw).hexdigest(),
            "process_identity_sha256": hashlib.sha256(process_raw).hexdigest(),
            "terminal_sha256": hashlib.sha256(terminal_raw).hexdigest(),
            "traffic_sha256": hashlib.sha256(traffic_raw).hexdigest(),
            "container_env_sha256": hashlib.sha256(container_env_raw).hexdigest(),
            "engine_ledger_sha256": hashlib.sha256(engine_raw).hexdigest(),
            "docker_log_sha256": hashlib.sha256(docker_raw).hexdigest(),
        },
    )
    print(json.dumps(json.loads(output.read_text(encoding="ascii")), sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("source-manifest")
    manifest.add_argument("--repo", required=True)
    manifest.add_argument("--source-commit", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=write_source_manifest)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--arm-dir", required=True)
    validate.add_argument("--source-commit", required=True)
    validate.add_argument("--task-id", required=True)
    validate.add_argument("--manifest-launch", required=True)
    validate.add_argument("--manifest-end", required=True)
    validate.add_argument("--output", required=True)
    validate.set_defaults(function=validate_gate)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except GateError as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
