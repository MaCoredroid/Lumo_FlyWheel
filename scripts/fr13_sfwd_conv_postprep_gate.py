#!/usr/bin/env python3
"""Issue and validate the source-bound SFWD conv/post-prep byte gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


CANDIDATE = "fixed32_sfwd_conv_postprep_frontier5_direct_v1"
SOURCE_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.source_manifest.v1"
READINESS_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.host_readiness.v1"
PASS_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.live_pass.v1"
TASK_ID = "astropy__astropy-12907"
TASK_MARKER = f"swe_verified:{TASK_ID}"
SUBSET = "config/fr13_fixed32/subset_b1_diagnostic_one.json"
SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
BLOCKS = "scripts/fr13_dvk_subset_blocks.json"
BLOCKS_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
QROW16_SHA256 = "1649fbe9c6886147710dc9be97567bffcac36175c26742b752be9be50c2cbb86"
QROW16_BYTES = 299_507_792
QROW16_PASS = (
    "results/fr13_fixed32_qrow16_num_splits0_live_pass_20260731T173608Z/"
    "fr13_fa2_qrow16_live_paged_ab.json"
)
QROW16_PASS_SHA256 = (
    "36940fd43d11399529d1bfe7e11baa9961907193267f3bb43d41057328737b77"
)
BYTE_SURFACES = (
    "query_spec",
    "key_spec",
    "value_spec",
    "value_tree",
    "g",
    "beta",
    "commit_source_stage",
)
SOURCE_FILES = (
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_b1_composed_stack_gate.py",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_cutlass_streamk_pass.py",
    "scripts/fr13_cutlass_wave_binary.py",
    "scripts/fr13_generate_sfwd_conv_postprep_fusion_kernel.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_patch_cutlass_fixed32_wave.py",
    "scripts/fr13_run_b1_cutlass_streamk_live_gate.sh",
    "scripts/fr13_run_b1_kernel_live_gate.sh",
    "scripts/fr13_run_b1_sfwd_conv_postprep_gate.sh",
    "scripts/fr13_run_b1_target_sfwd_conv_postprep_live_gate.sh",
    "scripts/fr13_sfwd_conv_postprep_gate.py",
    "scripts/run_swe_bench_q36_a.py",
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py",
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py",
    "src/lumo_flywheel_serving/fr13_sfwd_prior_reuse_descriptorless.py",
)
MODULE_SOURCE = "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion.py"
KERNEL_SOURCE = (
    "src/lumo_flywheel_serving/fr13_sfwd_conv_postprep_fusion_kernel.py"
)
LAYERS = 48


class GateError(RuntimeError):
    """A source, runtime, or evidence contract failed closed."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular(path: Path, *, nonempty: bool = True) -> bytes:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise GateError(f"required artifact is unreadable: {path}: {error}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or (nonempty and not raw)
    ):
        raise GateError(f"required artifact is not one regular file: {path}")
    return raw


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise GateError(f"artifact is not strict ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise GateError(f"artifact is not one JSON object: {path}")
    return payload, raw


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


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(repo: Path, *arguments: str) -> str:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(repo), *arguments],
                check=True,
                capture_output=True,
            )
            .stdout.decode("ascii")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as error:
        raise GateError(f"Git command failed: {' '.join(arguments)}") from error


def _file_record(path: Path) -> dict[str, Any]:
    raw = _regular(path)
    return {"bytes": len(raw), "sha256": _sha256(raw)}


def _stream_record(path: Path) -> dict[str, Any]:
    try:
        info = path.lstat()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise GateError(f"runtime asset is unreadable: {path}: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise GateError(f"runtime asset is not one regular file: {path}")
    return {"bytes": info.st_size, "sha256": digest.hexdigest()}


def _validate_source_manifest(
    payload: dict[str, Any], *, source_commit: str
) -> dict[str, dict[str, Any]]:
    files = payload.get("files")
    if (
        payload.get("schema") != SOURCE_SCHEMA
        or payload.get("candidate") != CANDIDATE
        or payload.get("source_commit") != source_commit
        or not isinstance(files, dict)
        or tuple(sorted(files)) != tuple(sorted(SOURCE_FILES))
    ):
        raise GateError("SFWD conv/post-prep source-manifest contract mismatch")
    for relative, entry in files.items():
        if (
            not isinstance(relative, str)
            or not isinstance(entry, dict)
            or type(entry.get("bytes")) is not int
            or entry["bytes"] <= 0
            or re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))
            is None
        ):
            raise GateError("SFWD conv/post-prep source-manifest entry is invalid")
    return files


def write_source_manifest(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GateError("source commit must be a full lowercase Git SHA")
    if _git(repo, "rev-parse", "HEAD") != source_commit:
        raise GateError("source commit does not equal repository HEAD")
    files: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_FILES:
        raw = _regular(repo / relative)
        try:
            committed = subprocess.run(
                ["git", "-C", str(repo), "show", f"{source_commit}:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as error:
            raise GateError(f"source file is not tracked: {relative}") from error
        if committed != raw:
            raise GateError(f"working source differs from commit: {relative}")
        files[relative] = {"bytes": len(raw), "sha256": _sha256(raw)}
    _write_json(
        Path(args.output).resolve(),
        {
            "schema": SOURCE_SCHEMA,
            "candidate": CANDIDATE,
            "source_commit": source_commit,
            "files": files,
        },
    )


def _validate_qrow_live_pass(repo: Path) -> None:
    path = repo / QROW16_PASS
    payload, raw = _load_json(path)
    if _sha256(raw) != QROW16_PASS_SHA256:
        raise GateError("pinned Qrow16 live PASS SHA-256 drifted")
    try:
        sys.path.insert(0, str(repo))
        from scripts import fr13_qrow16_pass_sidecar as qrow16

        qrow16.validate_live_result(payload, candidate_sha256=QROW16_SHA256)
    except (ImportError, ValueError) as error:
        raise GateError("pinned Qrow16 live PASS semantics drifted") from error


def write_host_readiness(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    if Path(_git(repo, "rev-parse", "--show-toplevel")).resolve() != repo:
        raise GateError("readiness repository is not the exact worktree root")
    if _git(repo, "rev-parse", "HEAD") != source_commit:
        raise GateError("readiness source commit does not equal HEAD")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise GateError("tracked worktree must be clean for readiness")
    branch = _git(repo, "symbolic-ref", "--short", "HEAD")
    upstream_commit = _git(repo, "rev-parse", "@{upstream}")
    if upstream_commit != source_commit:
        raise GateError("readiness source commit is not pushed to its upstream")

    manifest_path = Path(args.source_manifest).resolve()
    manifest, manifest_raw = _load_json(manifest_path)
    files = _validate_source_manifest(manifest, source_commit=source_commit)
    with tempfile.TemporaryDirectory(prefix="fr13-sfwd-conv-postprep-") as temporary:
        regenerated = Path(temporary) / "manifest.json"
        write_source_manifest(
            argparse.Namespace(
                repo=str(repo), source_commit=source_commit, output=str(regenerated)
            )
        )
        if _regular(regenerated) != manifest_raw:
            raise GateError("source manifest is not the exact regenerated manifest")

    fa2 = Path(args.fa2_so).resolve()
    fa2_record = _stream_record(fa2)
    if fa2_record != {"bytes": QROW16_BYTES, "sha256": QROW16_SHA256}:
        raise GateError("Qrow16 shared object size or SHA-256 drifted")
    if _file_record(repo / SUBSET)["sha256"] != SUBSET_SHA256:
        raise GateError("canonical one-task subset SHA-256 drifted")
    if _file_record(repo / BLOCKS)["sha256"] != BLOCKS_SHA256:
        raise GateError("K64 block-map SHA-256 drifted")
    _validate_qrow_live_pass(repo)
    manifest_sha = _sha256(manifest_raw)
    _write_json(
        Path(args.output).resolve(),
        {
            "schema": READINESS_SCHEMA,
            "status": "ready_for_one_real_swe_verified_hydra27_b1_byte_gate",
            "candidate": CANDIDATE,
            "source_commit": source_commit,
            "branch": branch,
            "upstream_commit": upstream_commit,
            "source_manifest_sha256": manifest_sha,
            "source_file_count": len(SOURCE_FILES),
            "candidate_source_sha256": files[MODULE_SOURCE]["sha256"],
            "candidate_kernel_source_sha256": files[KERNEL_SOURCE]["sha256"],
            "task_id": TASK_ID,
            "task_subset_sha256": SUBSET_SHA256,
            "draft_vocab_root": 1,
            "draft_vocab_k": 65536,
            "draft_vocab_blocks_sha256": BLOCKS_SHA256,
            "fixed32_mode": "hydra27_fixed32",
            "batch_size": 1,
            "physical_rows_per_request": 32,
            "qrow16_production": True,
            "qrow16_fa2_sha256": QROW16_SHA256,
            "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
            "compared_byte_surfaces": list(BYTE_SURFACES),
            "required_layer_count": LAYERS,
            "reference_always_served": True,
            "candidate_returned": False,
            "timing_eligible": False,
            "floor_acceptance_eligible": False,
            "production_eligible": False,
            "gpu_or_docker_used": False,
            "launched": False,
        },
    )


def _validate_live_pass(
    payload: dict[str, Any],
    *,
    source_commit: str,
    manifest_sha256: str,
) -> set[tuple[str, str]]:
    required = {
        "schema": PASS_SCHEMA,
        "status": "byte_pass_source_only",
        "candidate": CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": manifest_sha256,
        "fixed32_mode": "hydra27_fixed32",
        "task_marker": TASK_MARKER,
        "batch_size": 1,
        "task_count": 1,
        "layer_count": LAYERS,
        "physical_rows_per_request": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCKS_SHA256,
        "qrow16_production": True,
        "qrow16_fa2_sha256": QROW16_SHA256,
        "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
        "compared_byte_surfaces": list(BYTE_SURFACES),
        "real_task_authenticated": True,
        "reference_always_served": True,
        "candidate_returned": False,
        "reference_decision": "serve_incumbent",
        "candidate_decision": "shadow_only",
        "decision_exact": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "comparisons": LAYERS * len(BYTE_SURFACES),
        "mismatches": 0,
        "differing_bytes": 0,
        "errors": 0,
    }
    drift = {
        key: (payload.get(key), value)
        for key, value in required.items()
        if payload.get(key) != value
    }
    if drift:
        raise GateError(f"SFWD conv/post-prep live PASS drifted: {drift!r}")
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != LAYERS:
        raise GateError("SFWD conv/post-prep live PASS does not cover 48 layers")
    pairs: set[tuple[str, str]] = set()
    keys: set[str] = set()
    for entry in layers:
        if not isinstance(entry, dict):
            raise GateError("SFWD conv/post-prep live PASS layer is malformed")
        key = entry.get("layer_key")
        prefix = entry.get("layer_prefix_sha256")
        if (
            not isinstance(key, str)
            or re.fullmatch(r"0x[0-9a-f]+", key) is None
            or not isinstance(prefix, str)
            or re.fullmatch(r"[0-9a-f]{64}", prefix) is None
        ):
            raise GateError("SFWD conv/post-prep live PASS layer identity drifted")
        keys.add(key)
        pairs.add((key, prefix))
    if len(keys) != LAYERS or len(pairs) != LAYERS:
        raise GateError("SFWD conv/post-prep live PASS layer identities collide")
    return pairs


def validate_pass(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    manifest, manifest_raw = _load_json(Path(args.source_manifest).resolve())
    files = _validate_source_manifest(manifest, source_commit=source_commit)
    if _git(repo, "rev-parse", "HEAD") != source_commit:
        raise GateError("production source commit does not equal HEAD")
    for relative in SOURCE_FILES:
        if _file_record(repo / relative) != files[relative]:
            raise GateError(f"production source closure drifted: {relative}")
    live_pass, live_raw = _load_json(Path(args.live_pass).resolve())
    expected_pass_sha = str(args.expected_live_pass_sha256)
    expected_manifest_sha = str(args.expected_source_manifest_sha256)
    if _sha256(live_raw) != expected_pass_sha:
        raise GateError("SFWD conv/post-prep live PASS SHA-256 drifted")
    if _sha256(manifest_raw) != expected_manifest_sha:
        raise GateError("SFWD conv/post-prep source-manifest SHA-256 drifted")
    _validate_live_pass(
        live_pass,
        source_commit=source_commit,
        manifest_sha256=expected_manifest_sha,
    )


def _validate_qrow_evidence(logs: Path) -> tuple[bytes, bytes]:
    sidecar, sidecar_raw = _load_json(logs / "fr13_fa2_qrow16_production_pass.json")
    capture, capture_raw = _load_json(logs / "fr13_fa2_qrow16_production_capture.json")
    sidecar_sha = _sha256(sidecar_raw)
    if (
        sidecar.get("schema") != "fr13.fixed32.fa2_qrow16_production_pass.v1"
        or sidecar.get("status") != "PASS"
        or sidecar.get("candidate_so_sha256") != QROW16_SHA256
        or sidecar.get("live_result_sha256") != QROW16_PASS_SHA256
        or capture.get("schema")
        != "fr13.fixed32.fa2_qrow16_eager_production_engagement.v1"
        or capture.get("status") != "ENGAGED"
        or capture.get("runtime_mode") != "EAGER"
        or capture.get("batch_size") != 1
        or capture.get("layer_count") != 16
        or capture.get("candidate_so_sha256") != QROW16_SHA256
        or capture.get("pass_sidecar_sha256") != sidecar_sha
        or capture.get("dispatch") != "qrow16 exact geometry; no fallback"
        or capture.get("sfwd_state_fusion_production") is not False
        or capture.get("sfwd_conv_postprep_byte_ab") is not True
    ):
        raise GateError("pinned Qrow16 production evidence drifted")
    layers = capture.get("layers")
    if (
        not isinstance(layers, list)
        or len(layers) != 16
        or len(set(layers)) != 16
    ):
        raise GateError("Qrow16 eager engagement does not bind 16 unique layers")
    return sidecar_raw, capture_raw


def validate_gate(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        raise GateError(f"repository root is not a directory: {repo}")
    arm = Path(args.arm_dir).resolve()
    root = arm.parent
    logs = arm / "logs"
    source_commit = str(args.source_commit)
    task_id = str(args.task_id)
    if task_id != TASK_ID:
        raise GateError("gate task is not the canonical one-task identity")
    launch, launch_raw = _load_json(Path(args.manifest_launch).resolve())
    end, end_raw = _load_json(Path(args.manifest_end).resolve())
    if launch_raw != end_raw or launch != end:
        raise GateError("source manifest changed during the task")
    files = _validate_source_manifest(launch, source_commit=source_commit)
    source_sha = _sha256(launch_raw)

    readiness, readiness_raw = _load_json(
        root / "sfwd_conv_postprep_host_readiness.json"
    )
    readiness_required = {
        "schema": READINESS_SCHEMA,
        "status": "ready_for_one_real_swe_verified_hydra27_b1_byte_gate",
        "candidate": CANDIDATE,
        "source_commit": source_commit,
        "upstream_commit": source_commit,
        "source_manifest_sha256": source_sha,
        "source_file_count": len(SOURCE_FILES),
        "candidate_source_sha256": files[MODULE_SOURCE]["sha256"],
        "candidate_kernel_source_sha256": files[KERNEL_SOURCE]["sha256"],
        "task_id": TASK_ID,
        "task_subset_sha256": SUBSET_SHA256,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCKS_SHA256,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "qrow16_production": True,
        "qrow16_fa2_sha256": QROW16_SHA256,
        "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
        "compared_byte_surfaces": list(BYTE_SURFACES),
        "required_layer_count": LAYERS,
        "reference_always_served": True,
        "candidate_returned": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "gpu_or_docker_used": False,
        "launched": False,
    }
    if any(readiness.get(key) != value for key, value in readiness_required.items()):
        raise GateError("SFWD conv/post-prep host-readiness contract drifted")

    runtime_launch = _regular(root / "runtime_manifest.at_launch.json")
    runtime_end = _regular(root / "runtime_manifest.at_end.json")
    external_launch = _regular(root / "external_manifest.at_launch.json")
    external_end = _regular(root / "external_manifest.at_end.json")
    if runtime_launch != runtime_end or external_launch != external_end:
        raise GateError("runtime or external manifest changed during the task")

    installed = logs / "fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
    installed_raw = _regular(installed)
    if (
        installed_raw != launch_raw
        or stat.S_IMODE(installed.lstat().st_mode) != 0o400
    ):
        raise GateError("installed source manifest is not launch-bound")
    marker = logs / "fr13_fixed32_sfwd_state_fusion.real_event.arm"
    if (
        _regular(marker) != (TASK_MARKER + "\n").encode("ascii")
        or stat.S_IMODE(marker.lstat().st_mode) != 0o444
    ):
        raise GateError("authenticated real-event marker drifted")

    live_pass, pass_raw = _load_json(
        logs / "fr13_fixed32_sfwd_conv_postprep.live_pass.json"
    )
    pass_pairs = _validate_live_pass(
        live_pass, source_commit=source_commit, manifest_sha256=source_sha
    )
    records_raw = _regular(
        logs / "fr13_fixed32_sfwd_conv_postprep.byte_ab.jsonl"
    )
    try:
        records = [
            json.loads(
                line,
                object_pairs_hook=_reject_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(value)
                ),
            )
            for line in records_raw.decode("ascii").splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise GateError("SFWD conv/post-prep comparison records are malformed") from error
    if not records or any(not isinstance(record, dict) for record in records):
        raise GateError("SFWD conv/post-prep byte gate was vacuous")
    record_pairs: set[tuple[str, str]] = set()
    for record in records:
        expected = {
            "schema": "fr13.fixed32.sfwd_conv_postprep.byte_ab.v1",
            "status": "pass",
            "candidate": CANDIDATE,
            "source_commit": source_commit,
            "source_manifest_sha256": source_sha,
            "fixed32_mode": "hydra27_fixed32",
            "task_marker": TASK_MARKER,
            "batch_size": 1,
            "physical_rows_per_request": 32,
            "compared_byte_surfaces": list(BYTE_SURFACES),
            "mismatches": 0,
            "differing_bytes": 0,
            "zero_diff": True,
            "real_task_authenticated": True,
            "reference_always_served": True,
            "candidate_returned": False,
            "reference_decision": "serve_incumbent",
            "candidate_decision": "shadow_only",
            "decision_exact": True,
            "qrow16_production": True,
            "qrow16_fa2_sha256": QROW16_SHA256,
            "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
            "timing_eligible": False,
            "floor_acceptance_eligible": False,
            "production_eligible": False,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise GateError("SFWD conv/post-prep comparison record drifted")
        comparisons = record.get("comparisons")
        if (
            not isinstance(comparisons, list)
            or [item.get("name") for item in comparisons] != list(BYTE_SURFACES)
            or any(
                item.get("byte_equal") is not True
                or item.get("differing_bytes") != 0
                for item in comparisons
                if isinstance(item, dict)
            )
            or any(not isinstance(item, dict) for item in comparisons)
        ):
            raise GateError("comparison surfaces are incomplete or unequal")
        record_pairs.add(
            (str(record.get("layer_key")), str(record.get("layer_prefix_sha256")))
        )
    if len(record_pairs) != LAYERS or record_pairs != pass_pairs:
        raise GateError("comparison records do not bind the PASS layer set")

    qrow_sidecar, qrow_capture = _validate_qrow_evidence(logs)
    diagnostic, diagnostic_raw = _load_json(arm / "fixed32_b1_diagnostic.json")
    terminal, terminal_raw = _load_json(arm / "fixed32_final_flush_skipped.json")
    traffic, traffic_raw = _load_json(
        arm / "fixed32_chat_traffic_audit_skipped.json"
    )
    if (
        diagnostic.get("task_ids") != [TASK_ID]
        or diagnostic.get("floor_acceptance_eligible") is not False
        or terminal
        != {
            "schema": "fr13-fixed32-eager-kernel-terminal-v1",
            "run_classification": "eager_kernel_byte_diagnostic",
            "acceptance_valid": False,
            "flush_protocol_used": False,
        }
        or traffic.get("run_classification") != "eager_kernel_byte_diagnostic"
        or traffic.get("acceptance_valid") is not False
    ):
        raise GateError("one-task eager diagnostic evidence drifted")
    brackets = list(
        arm.glob(f"swe_out/*/per_task/{TASK_ID}/fixed32_task_boundary.json")
    )
    if len(brackets) != 1:
        raise GateError("one-task boundary evidence is missing or ambiguous")
    bracket, bracket_raw = _load_json(brackets[0])
    if (
        bracket.get("schema")
        != "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1"
        or bracket.get("instance_id") != TASK_ID
        or bracket.get("acceptance_valid") is not False
    ):
        raise GateError("one-task boundary contract drifted")

    container_env_raw = _regular(arm / "container_env.txt")
    container_env = container_env_raw.decode("ascii").splitlines()
    target_selector = "FR13_FIXED32_CUTLASS_WAVE=identity_wide256_fullgrid_b1_byte_ab"
    target_live_sha256: str | None = None
    if target_selector in container_env:
        if not args.target_live_pass or not args.target_candidate_so:
            raise GateError("combined SFWD credential is missing its target-GEMM PASS")
        target_path = Path(args.target_live_pass).resolve()
        target_raw = _regular(target_path)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import fr13_cutlass_streamk_pass as cutlass_pass

        try:
            cutlass_pass.validate_live_result(
                target_path,
                _sha256(target_raw),
                Path(args.target_candidate_so).resolve(),
                repo / "scripts/fr13_patch_cutlass_fixed32_wave.py",
                expected_source_commit=source_commit,
                candidate_selector="identity_wide256_fullgrid_b1",
                qualification_profile="k64_root",
                diagnostic_task_profile="astropy12907",
            )
        except (OSError, ValueError, cutlass_pass.QualificationError) as error:
            raise GateError("combined target-GEMM PASS drifted") from error
        target_live_sha256 = _sha256(target_raw)
    elif args.target_live_pass or args.target_candidate_so:
        raise GateError("target-GEMM evidence was supplied to an isolated SFWD gate")
    expected_env = (
        "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=0",
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=1",
        f"FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_MANIFEST_SHA256={source_sha}",
        f"FR13_FIXED32_SFWD_CONV_POSTPREP_SOURCE_COMMIT={source_commit}",
        "FR13_FIXED32_SFWD_STATE_FUSION_BYTE_AB=0",
        "FR13_FIXED32_SFWD_STATE_FUSION_PRODUCTION=0",
        "FR13_FIXED32_SFWD_PRIOR_REUSE_BYTE_AB=0",
        "FR13_FA2_QROW16_PRODUCTION=1",
        f"FR13_FA2_QROW16_SO_SHA256={QROW16_SHA256}",
        f"FR13_FA2_QROW16_LIVE_PASS_SHA256={QROW16_PASS_SHA256}",
        "FR13_DRAFT_VOCAB_ROOT=1",
        "FR13_DRAFT_VOCAB_K=65536",
        "ENFORCE_EAGER=1",
    )
    if any(container_env.count(value) != 1 for value in expected_env):
        raise GateError("container environment did not preserve the exclusive gate")
    if "FR10_ALLOW_LINEAR_FALLBACK=1" in container_env:
        raise GateError("linear fallback was enabled")
    docker_raw = _regular(arm / "docker_after_tasks.log")
    docker_text = docker_raw.decode("utf-8", errors="replace")
    if (
        docker_text.count("[FR13_DRAFT_VOCAB] shim built K=65536 ") != 1
        or docker_text.count("[FR13_DRAFT_VOCAB_ROOT] engaged K=65536 ") != 1
        or "linear fallback engaged" in docker_text
    ):
        raise GateError("K64/root1 engagement or no-fallback evidence drifted")

    output = Path(args.output).resolve()
    _write_json(
        output,
        {
            "schema": "fr13.fixed32.sfwd_conv_postprep.k64_root_b1_gate.v1",
            "status": "pass",
            "candidate": CANDIDATE,
            "source_commit": source_commit,
            "source_manifest_sha256": source_sha,
            "task_id": TASK_ID,
            "task_count": 1,
            "fixed32_mode": "hydra27_fixed32",
            "batch_size": 1,
            "physical_rows_per_request": 32,
            "draft_vocab_root": 1,
            "draft_vocab_k": 65536,
            "qrow16_production": True,
            "qrow16_fa2_sha256": QROW16_SHA256,
            "qrow16_live_pass_sha256": QROW16_PASS_SHA256,
            "layer_count": LAYERS,
            "compared_byte_surfaces": list(BYTE_SURFACES),
            "reference_returned": True,
            "candidate_returned": False,
            "decision_exact": True,
            "no_fallback": True,
            "timing_eligible": False,
            "floor_acceptance_eligible": False,
            "production_enabled": False,
            "records_sha256": _sha256(records_raw),
            "live_pass_sha256": _sha256(pass_raw),
            "runtime_manifest_sha256": _sha256(runtime_launch),
            "external_manifest_sha256": _sha256(external_launch),
            "host_readiness_sha256": _sha256(readiness_raw),
            "diagnostic_sha256": _sha256(diagnostic_raw),
            "task_bracket_sha256": _sha256(bracket_raw),
            "terminal_sha256": _sha256(terminal_raw),
            "traffic_sha256": _sha256(traffic_raw),
            "container_env_sha256": _sha256(container_env_raw),
            "docker_log_sha256": _sha256(docker_raw),
            "qrow16_sidecar_sha256": _sha256(qrow_sidecar),
            "qrow16_capture_sha256": _sha256(qrow_capture),
            "combined_target_selector": (
                "identity_wide256_fullgrid_b1" if target_live_sha256 else None
            ),
            "combined_target_live_pass_sha256": target_live_sha256,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("source-manifest")
    manifest.add_argument("--repo", required=True)
    manifest.add_argument("--source-commit", required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=write_source_manifest)
    readiness = commands.add_parser("host-readiness")
    readiness.add_argument("--repo", required=True)
    readiness.add_argument("--source-commit", required=True)
    readiness.add_argument("--source-manifest", required=True)
    readiness.add_argument("--fa2-so", required=True)
    readiness.add_argument("--output", required=True)
    readiness.set_defaults(function=write_host_readiness)
    pass_parser = commands.add_parser("validate-pass")
    pass_parser.add_argument("--repo", required=True)
    pass_parser.add_argument("--live-pass", required=True)
    pass_parser.add_argument("--expected-live-pass-sha256", required=True)
    pass_parser.add_argument("--source-manifest", required=True)
    pass_parser.add_argument("--expected-source-manifest-sha256", required=True)
    pass_parser.add_argument("--source-commit", required=True)
    pass_parser.set_defaults(function=validate_pass)
    validate = commands.add_parser("validate")
    validate.add_argument("--repo", required=True)
    validate.add_argument("--arm-dir", required=True)
    validate.add_argument("--source-commit", required=True)
    validate.add_argument("--task-id", required=True)
    validate.add_argument("--manifest-launch", required=True)
    validate.add_argument("--manifest-end", required=True)
    validate.add_argument("--target-live-pass")
    validate.add_argument("--target-candidate-so")
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
