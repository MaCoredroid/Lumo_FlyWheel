#!/usr/bin/env python3
"""Validate the shared B1 graph gate and issue its scoped credentials."""

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

import fr13_fixed32_contract as fixed32_contract
import fr13_cfwd_logit_direct_gate as cfwd_gate
import fr13_qrow32_b1_pass_sidecar as qrow32
import fr13_runtime_manifest as runtime_manifest


TASK_ID = "astropy__astropy-12907"
SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
BLOCK_MAP_SHA256 = "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
DFWD_SCHEMA = "fr13.fixed32.dfwd_k64_top3.real_task_credential.v1"
DFWD_BUILD_SCHEMA = "fr13.fixed32.dfwd_k64_mapped_top3_sm121a_build.v1"
DFWD_CANDIDATE_SHA256 = (
    "c0ed75cafdd926eceafcf28671869d54f37addb51bfef5a37c0b07c34f5420ff"
)
DFWD_CANDIDATE_BYTES = 159_288
DFWD_SOURCE = "csrc/fr13_dfwd_k64_top3.cu"
DFWD_SOURCE_SHA256 = (
    "38ea96d955355bf172f534174aa2d91e6db23170144b1c84c9474016a6c05e72"
)
GQA3_CANDIDATE = "fixed32_gdn_single_launch_gqa_group3_v1"
QROW_CREDENTIAL_SCHEMA = (
    "fr13.fixed32.fa2_qrow32_split2_k64_b1_live_verification.v2"
)
SFWD_COMBINED_SCHEMA = "fr13.fixed32.sfwd_conv_postprep.k64_root_b1_gate.v1"
SFWD_CANDIDATE = "fixed32_sfwd_conv_postprep_frontier5_direct_v1"
PRODUCTION_SMOKE_SCHEMA = "fr13.fixed32.b1_composed_cfwd.production_smoke.v1"
TARGET_SELECTOR = "identity_wide256_fullgrid_b1"
TARGET_SHA256 = "85937b5c35ec87bce12e4b5d677dd67f63004f9a9d9fb6d64473a5bd3b53b2da"
CFWD_ENGAGEMENT_SCHEMA = "fr13.fixed32.cfwd_logit_direct.production_engagement.v1"
CFWD_SERVED_RETURN = "logit-direct candidate products"
DFWD_MARKERS = (
    "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3",
    "[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0",
    "[FR13_DFWD_K64_TOP3] graph captured_calls=4",
)
SFWD_MARKER = "[FR13_SFWD_CONV_POSTPREP] production engaged layer="
SIX_WAY_ENV = (
    "FR13_FIXED32_MODE=hydra27_fixed32",
    "FR13_FIXED32_B1_DIAGNOSTIC=0",
    "FR13_DRAFT_VOCAB_ROOT=1",
    "FR13_DRAFT_VOCAB_K=65536",
    "MAX_NUM_SEQS=1",
    "SWE_CONCURRENCY=1",
    "ENFORCE_EAGER=0",
    "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
    "FR13_FA2_QROW16_LIVE_PAGED_AB=0",
    "FR13_FA2_QROW16_PRODUCTION=0",
    "FR13_FA2_QROW32_B1_LIVE_AB_ARM=",
    "FR13_FA2_QROW32_B1_PRODUCTION_ARM=split2",
    "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=1",
    "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=1",
    "FR13_DFWD_K64_TOP3=1",
    f"FR13_DFWD_K64_TOP3_SHA256={DFWD_CANDIDATE_SHA256}",
    f"FR13_FIXED32_CUTLASS_WAVE={TARGET_SELECTOR}",
    "FR13_FIXED32_CUTLASS_WAVE_PRODUCTION=1",
    "FR13_FIXED32_SFWD_CONV_POSTPREP_FUSION=1",
    "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB=0",
    "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE=0",
    "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=1",
    "FR13_CFWD_LOGIT_DIRECT_BYTE_AB=0",
    "FR13_CFWD_LOGIT_DIRECT_PRODUCTION=1",
)
COMPONENT_ARGUMENTS = (
    ("qrow32_composed", "qrow_credential", "qrow_credential_sha256"),
    ("gqa3", "gdn_credential", "gdn_credential_sha256"),
    ("dfwd_top3", "dfwd_credential", "dfwd_credential_sha256"),
    ("target_live", "target_live", "target_live_sha256"),
    ("sfwd_pass", "sfwd_pass", "sfwd_pass_sha256"),
    ("sfwd_manifest", "source_manifest", "source_manifest_sha256"),
    ("target_sfwd_summary", "combined_summary", "combined_summary_sha256"),
    ("taw_b1_credential", "taw_b1_credential", "taw_b1_credential_sha256"),
    ("taw_b1_live_bundle", "taw_b1_live_bundle", "taw_b1_live_bundle_sha256"),
    ("taw_b4_pass", "taw_b4_pass", "taw_b4_pass_sha256"),
    ("taw_b4_verdict", "taw_b4_verdict", "taw_b4_verdict_sha256"),
    ("taw_merge_binding", "taw_merge_binding", "taw_merge_binding_sha256"),
    ("taw_production", "taw_production", "taw_production_sha256"),
    ("cfwd_credential", "cfwd_credential", "cfwd_credential_sha256"),
)


class GateError(RuntimeError):
    """A composed-gate input or runtime artifact failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular(path: Path) -> bytes:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise GateError(f"required artifact is unreadable: {path}: {error}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not raw
    ):
        raise GateError(f"required artifact is not one regular file: {path}")
    return raw


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                GateError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
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
    os.chmod(temporary, 0o400)
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise GateError(f"Git command failed: {' '.join(arguments)}") from error


def _validate_source_commit(repo: Path, source_commit: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise GateError("source commit must be one full lowercase Git SHA")
    if _git(repo, "rev-parse", "HEAD").decode("ascii").strip() != source_commit:
        raise GateError("source commit does not equal repository HEAD")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=no"):
        raise GateError("tracked worktree is not clean")


def _validate_build(
    *,
    repo: Path,
    source_commit: str,
    candidate_so: Path,
    build_attestation: Path,
) -> tuple[dict[str, Any], bytes, bytes]:
    _validate_source_commit(repo, source_commit)
    candidate_raw = _regular(candidate_so)
    if (
        len(candidate_raw) != DFWD_CANDIDATE_BYTES
        or _sha256(candidate_raw) != DFWD_CANDIDATE_SHA256
    ):
        raise GateError("DFWD K64 top3 candidate binary identity drifted")
    source_raw = _regular(repo / DFWD_SOURCE)
    committed_source = _git(repo, "show", f"{source_commit}:{DFWD_SOURCE}")
    if (
        source_raw != committed_source
        or _sha256(source_raw) != DFWD_SOURCE_SHA256
    ):
        raise GateError("DFWD K64 top3 source is not final-commit bound")
    attestation, attestation_raw = _load_json(build_attestation)
    binary = attestation.get("binary")
    source = attestation.get("source")
    contract = attestation.get("kernel_contract")
    if (
        attestation.get("schema") != DFWD_BUILD_SCHEMA
        or attestation.get("status") != "BUILT_UNQUALIFIED"
        or attestation.get("byte_equality_claim") is not False
        or attestation.get("real_task_correctness") is not False
        or not isinstance(binary, dict)
        or binary.get("bytes") != DFWD_CANDIDATE_BYTES
        or binary.get("sha256") != DFWD_CANDIDATE_SHA256
        or not isinstance(source, dict)
        or source.get("path") != DFWD_SOURCE
        or source.get("sha256") != DFWD_SOURCE_SHA256
        or not isinstance(contract, dict)
        or contract.get("minimum_reduction_launches_eliminated_per_event") != 45
        or contract.get("outputs") != ["int64[1] spine", "int64[1,3] mapped top3"]
    ):
        raise GateError("DFWD K64 top3 build attestation drifted")
    return attestation, attestation_raw, candidate_raw


def validate_dfwd_build(args: argparse.Namespace) -> None:
    attestation, attestation_raw, candidate_raw = _validate_build(
        repo=Path(args.repo).resolve(),
        source_commit=str(args.source_commit),
        candidate_so=Path(args.candidate_so).resolve(),
        build_attestation=Path(args.build_attestation).resolve(),
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "candidate_sha256": _sha256(candidate_raw),
                "build_attestation_sha256": _sha256(attestation_raw),
                "source_sha256": attestation["source"]["sha256"],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def validate_graph_credentials(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    _attestation, attestation_raw, candidate_raw = _validate_build(
        repo=repo,
        source_commit=source_commit,
        candidate_so=Path(args.candidate_so).resolve(),
        build_attestation=Path(args.build_attestation).resolve(),
    )
    qrow, qrow_raw = _load_json(Path(args.qrow_credential).resolve())
    dfwd, dfwd_raw = _load_json(Path(args.dfwd_credential).resolve())
    gdn_raw = _regular(Path(args.gdn_credential).resolve())
    qrow_live, qrow_live_raw = _load_json(Path(args.qrow_live).resolve())
    expected_hashes = {
        "Qrow32 composed credential": (
            qrow_raw,
            str(args.qrow_credential_sha256),
        ),
        "Qrow32 raw live PASS": (
            qrow_live_raw,
            str(args.qrow_live_sha256),
        ),
        "GQA3 credential": (gdn_raw, str(args.gdn_credential_sha256)),
        "DFWD top3 credential": (
            dfwd_raw,
            str(args.dfwd_credential_sha256),
        ),
    }
    for label, (raw, expected) in expected_hashes.items():
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise GateError(f"{label} SHA-256 is malformed")
        if _sha256(raw) != expected:
            raise GateError(f"{label} SHA-256 drifted")
    common = {
        "status": "PASS",
        "source_commit": source_commit,
        "task_id": TASK_ID,
        "task_count": 1,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "concurrency": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "gqa3_credential_sha256": _sha256(gdn_raw),
        "task_completion_verified": True,
        "authenticated_real_task_verified": True,
        "performance_measurement": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
    }
    for label, payload in (("Qrow32", qrow), ("DFWD top3", dfwd)):
        for key, expected in common.items():
            if type(payload.get(key)) is not type(expected) or payload.get(key) != expected:
                raise GateError(f"{label} credential {key} drifted")
    if (
        qrow.get("schema") != QROW_CREDENTIAL_SCHEMA
        or qrow.get("candidate_so_sha256") != qrow32.CANDIDATE_SHA256
        or qrow.get("candidate_so_size") != qrow32.CANDIDATE_SIZE
        or qrow.get("fa2_head") != qrow32.FA2_HEAD
        or qrow.get("fa2_source_closure_sha256") != qrow32.SOURCE_CLOSURE_SHA256
        or qrow.get("reference_served") is not True
        or qrow.get("raw_byte_equal") is not True
        or qrow.get("live_result_sha256") != _sha256(qrow_live_raw)
    ):
        raise GateError("Qrow32 composed credential contract drifted")
    try:
        qrow32.validate_live_result(
            qrow_live,
            candidate_sha256=qrow32.CANDIDATE_SHA256,
            arm=qrow32.ARM,
            source_commit=source_commit,
            patch_source_sha256=_sha256(
                _regular(repo / "scripts/fr13_patch_fa2_tree_bias.py")
            ),
        )
    except ValueError as error:
        raise GateError("Qrow32 raw live PASS contract drifted") from error
    if (
        dfwd.get("schema") != DFWD_SCHEMA
        or dfwd.get("candidate_so_sha256") != _sha256(candidate_raw)
        or dfwd.get("candidate_so_bytes") != len(candidate_raw)
        or dfwd.get("candidate_source") != DFWD_SOURCE
        or dfwd.get("candidate_source_sha256") != DFWD_SOURCE_SHA256
        or dfwd.get("build_attestation_sha256") != _sha256(attestation_raw)
        or dfwd.get("ready_marker_verified") is not True
        or dfwd.get("engaged_marker_verified") is not True
        or dfwd.get("full_graph_capture_marker_verified") is not True
        or dfwd.get("minimum_reduction_launches_eliminated_per_event") != 45
        or dfwd.get("drafter_byte_equality_claim") is not False
        or dfwd.get("target_verifier_and_rejection_sampling_remain_authoritative")
        is not True
    ):
        raise GateError("DFWD top3 credential contract drifted")
    shared_evidence = (
        "runtime_manifest_sha256",
        "external_manifest_sha256",
        "health_sha256",
        "traffic_audit_sha256",
        "container_env_sha256",
        "docker_log_sha256",
        "gqa3_credential_sha256",
    )
    for key in shared_evidence:
        value = qrow.get(key)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            or dfwd.get(key) != value
        ):
            raise GateError(f"Gate-A shared evidence {key} drifted")
    print(
        json.dumps(
            {
                "status": "ready",
                "source_commit": source_commit,
                "qrow_credential_sha256": _sha256(qrow_raw),
                "gqa3_credential_sha256": _sha256(gdn_raw),
                "dfwd_credential_sha256": _sha256(dfwd_raw),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def validate_eager_credentials(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    _validate_source_commit(repo, source_commit)
    summary, summary_raw = _load_json(Path(args.combined_summary).resolve())
    target_raw = _regular(Path(args.target_live).resolve())
    sfwd_raw = _regular(Path(args.sfwd_pass).resolve())
    manifest_raw = _regular(Path(args.source_manifest).resolve())
    checks = (
        ("combined target/SFWD summary", summary_raw, args.combined_summary_sha256),
        ("target live PASS", target_raw, args.target_live_sha256),
        ("SFWD live PASS", sfwd_raw, args.sfwd_pass_sha256),
        ("SFWD source manifest", manifest_raw, args.source_manifest_sha256),
    )
    for label, raw, expected_value in checks:
        expected = str(expected_value)
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected) is None
            or _sha256(raw) != expected
        ):
            raise GateError(f"{label} SHA-256 drifted")
    expected_summary = {
        "schema": SFWD_COMBINED_SCHEMA,
        "status": "pass",
        "candidate": SFWD_CANDIDATE,
        "source_commit": source_commit,
        "source_manifest_sha256": _sha256(manifest_raw),
        "task_id": TASK_ID,
        "task_count": 1,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "physical_rows_per_request": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "qrow16_production": True,
        "layer_count": 48,
        "reference_returned": True,
        "candidate_returned": False,
        "decision_exact": True,
        "no_fallback": True,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
        "live_pass_sha256": _sha256(sfwd_raw),
        "combined_target_selector": "identity_wide256_fullgrid_b1",
        "combined_target_live_pass_sha256": _sha256(target_raw),
    }
    for key, expected in expected_summary.items():
        if type(summary.get(key)) is not type(expected) or summary.get(key) != expected:
            raise GateError(f"Gate-B combined summary {key} drifted")
    for key in (
        "records_sha256",
        "runtime_manifest_sha256",
        "external_manifest_sha256",
        "host_readiness_sha256",
        "diagnostic_sha256",
        "task_bracket_sha256",
        "terminal_sha256",
        "traffic_sha256",
        "container_env_sha256",
        "docker_log_sha256",
        "qrow16_sidecar_sha256",
        "qrow16_capture_sha256",
    ):
        value = summary.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise GateError(f"Gate-B combined summary {key} is not a SHA-256")
    print(
        json.dumps(
            {
                "status": "ready",
                "source_commit": source_commit,
                "combined_summary_sha256": _sha256(summary_raw),
                "target_live_pass_sha256": _sha256(target_raw),
                "sfwd_live_pass_sha256": _sha256(sfwd_raw),
                "source_manifest_sha256": _sha256(manifest_raw),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def _validate_health_and_traffic(arm: Path) -> tuple[bytes, bytes]:
    health, health_raw = _load_json(arm / "health.json")
    tasks = health.get("tasks")
    if (
        health.get("swe_orchestrator_rc") != 0
        or not isinstance(tasks, list)
        or len(tasks) != 1
        or tasks[0].get("instance_id") != TASK_ID
        or tasks[0].get("codex_timed_out") is not False
        or tasks[0].get("verdict") != "resolved"
    ):
        raise GateError("real SWE-Verified task did not resolve cleanly")
    traffic, traffic_raw = _load_json(arm / "fixed32_chat_traffic_audit.json")
    checks = traffic.get("checks")
    subset = traffic.get("subset")
    if (
        traffic.get("schema")
        not in {
            "fr13-fixed32-chat-task-provenance-audit-v2",
            "fr13-fixed32-chat-task-provenance-audit-v3",
        }
        or traffic.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or traffic.get("mode") != "hydra27_fixed32"
        or not isinstance(subset, dict)
        or subset.get("sha256") != SUBSET_SHA256
        or subset.get("task_ids") != [TASK_ID]
        or subset.get("task_count") != 1
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
        or traffic.get("exact_proxy_engine_attempt_parity") is not True
        or set(traffic.get("tasks", {})) != {TASK_ID}
    ):
        raise GateError("authenticated real-task traffic audit drifted")
    return health_raw, traffic_raw


def _require_env(container_env: bytes, required: tuple[str, ...]) -> None:
    try:
        lines = container_env.decode("ascii").splitlines()
    except UnicodeError as error:
        raise GateError("container environment is not ASCII") from error
    missing = [value for value in required if lines.count(value) != 1]
    if missing:
        raise GateError(f"combined graph gate environment drifted: {missing!r}")


def validate_six_way_environment(container_env: bytes) -> None:
    """Require the one admitted Qrow32/CFWD production composition exactly."""
    _require_env(container_env, SIX_WAY_ENV)


def _expect_sha(raw: bytes, expected: object, label: str) -> str:
    value = str(expected)
    if re.fullmatch(r"[0-9a-f]{64}", value) is None or _sha256(raw) != value:
        raise GateError(f"{label} SHA-256 drifted")
    return value


def _component_hashes(args: argparse.Namespace) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, path_name, sha_name in COMPONENT_ARGUMENTS:
        raw = _regular(Path(getattr(args, path_name)).resolve())
        hashes[key] = _expect_sha(raw, getattr(args, sha_name), key)
    return hashes


def _validate_cfwd_engagement(
    payload: dict[str, Any], *, source_commit: str, credential_sha256: str
) -> None:
    expected_keys = {
        "schema",
        "status",
        "candidate",
        "mode",
        "batch_size",
        "source_commit",
        "candidate_source_sha256",
        "production_pass_sha256",
        "served_return",
        "producer_pid",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != CFWD_ENGAGEMENT_SCHEMA
        or payload.get("status") != "engaged"
        or payload.get("candidate") != cfwd_gate.CANDIDATE
        or payload.get("mode") != "hydra27_fixed32"
        or payload.get("batch_size") != 1
        or payload.get("source_commit") != source_commit
        or payload.get("candidate_source_sha256")
        != cfwd_gate.CANDIDATE_SOURCE_SHA256
        or payload.get("production_pass_sha256") != credential_sha256
        or payload.get("served_return") != CFWD_SERVED_RETURN
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
    ):
        raise GateError("CFWD production served-return engagement drifted")


def _validate_production_smoke_payload(
    payload: object,
    *,
    source_commit: str,
    component_hashes: dict[str, str],
) -> dict[str, Any]:
    keys = {
        "schema",
        "status",
        "source_commit",
        "mode",
        "batch_size",
        "task_id",
        "task_count",
        "subset_sha256",
        "runtime_mode",
        "production_paths",
        "component_credential_sha256s",
        "runtime_evidence_sha256s",
        "cfwd_served_return",
        "performance_measurement",
        "timing_eligible",
        "exact4_eligible",
    }
    paths = {
        "qrow32_split2": True,
        "gdn_gqa_group3": True,
        "dfwd_k64_top3": True,
        "target_gemm_selector": TARGET_SELECTOR,
        "sfwd_conv_postprep": True,
        "taw_native_precompute": True,
        "cfwd_logit_direct": True,
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != keys
        or payload.get("schema") != PRODUCTION_SMOKE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("source_commit") != source_commit
        or payload.get("mode") != "hydra27_fixed32"
        or payload.get("batch_size") != 1
        or payload.get("task_id") != TASK_ID
        or payload.get("task_count") != 1
        or payload.get("subset_sha256") != SUBSET_SHA256
        or payload.get("runtime_mode") != "FULL"
        or payload.get("production_paths") != paths
        or payload.get("component_credential_sha256s") != component_hashes
        or payload.get("cfwd_served_return") != CFWD_SERVED_RETURN
        or payload.get("performance_measurement") is not False
        or payload.get("timing_eligible") is not False
        or payload.get("exact4_eligible") is not True
    ):
        raise GateError("composed CFWD production smoke credential drifted")
    runtime_hashes = payload.get("runtime_evidence_sha256s")
    if (
        not isinstance(runtime_hashes, dict)
        or not runtime_hashes
        or any(
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in runtime_hashes.values()
        )
    ):
        raise GateError("composed CFWD production smoke evidence drifted")
    return payload


def issue_production_smoke(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    arm = Path(args.arm).resolve()
    source_commit = str(args.source_commit)
    _validate_source_commit(repo, source_commit)
    component_hashes = _component_hashes(args)

    runtime_launch = _regular(Path(args.runtime_launch).resolve())
    runtime_end = _regular(Path(args.runtime_end).resolve())
    external_launch = _regular(Path(args.external_launch).resolve())
    external_end = _regular(Path(args.external_end).resolve())
    if runtime_launch != runtime_end or external_launch != external_end:
        raise GateError("runtime or external manifest changed during production smoke")
    runtime, _ = _load_json(Path(args.runtime_end).resolve())
    try:
        regenerated = runtime_manifest.build_manifest(
            repo,
            profile="fixed32",
            sequence="scripts/fr13_fixed32_floor_timers_seq.sh",
            source_commit=source_commit,
        )
    except runtime_manifest.ManifestError as error:
        raise GateError(f"cannot regenerate runtime manifest: {error}") from error
    if runtime != regenerated:
        raise GateError("runtime manifest is not the canonical final-commit closure")
    external, _ = _load_json(Path(args.external_end).resolve())
    try:
        fixed32_contract.validate_external_manifest(external)
    except fixed32_contract.ContractError as error:
        raise GateError(f"external manifest is invalid: {error}") from error
    if _regular(arm / "git_head.txt") != f"{source_commit}\n".encode("ascii"):
        raise GateError("production smoke git_head.txt drifted")
    launcher_meta = _regular(arm.parent / "launcher_meta.txt").decode("ascii")
    for line in (
        f"source={source_commit}",
        "task_count=1",
        "timing_eligible=0",
        "cfwd_logit_direct_production=1",
    ):
        if launcher_meta.splitlines().count(line) != 1:
            raise GateError(f"production smoke launcher metadata drifted: {line}")

    health_raw, traffic_raw = _validate_health_and_traffic(arm)
    container_raw = _regular(arm / "container_env.txt")
    validate_six_way_environment(container_raw)
    docker_raw = _regular(arm / "docker_after_tasks.log")
    docker_text = docker_raw.decode("utf-8", errors="replace")
    if any(marker not in docker_text for marker in DFWD_MARKERS):
        raise GateError("DFWD K64 top3 production evidence is incomplete")
    layers = set(
        re.findall(
            r"\[FR13_SFWD_CONV_POSTPREP\] production engaged "
            r"layer=([^ ]+) B=1 rows=32",
            docker_text,
        )
    )
    if len(layers) != 48 or docker_text.count(SFWD_MARKER) != 48:
        raise GateError("SFWD conv/post-prep did not engage exactly 48 layers")

    gqa_raw = _regular(
        arm / "logs/fr13_fixed32_gdn_gqa_group3.production_credential.json"
    )
    _expect_sha(gqa_raw, component_hashes["gqa3"], "copied GQA3 credential")
    if (
        _regular(arm / "logs/fr13_fixed32_gdn_gqa_group3.production.arm") != b"1\n"
        or _regular(
            arm / "logs/fr13_fixed32_gdn_gqa_group3.production_batch.flag"
        )
        != b"1\n"
    ):
        raise GateError("GQA3 production arm or batch marker drifted")
    if _regular(arm / "logs/fr13_fixed32_cutlass_wave.selector") != (
        TARGET_SELECTOR + "\n"
    ).encode("ascii"):
        raise GateError("target GEMM selector record drifted")
    sfwd_pass_raw = _regular(
        arm / "logs/fr13_fixed32_sfwd_conv_postprep.production_pass.json"
    )
    sfwd_manifest_raw = _regular(
        arm / "logs/fr13_fixed32_sfwd_conv_postprep.source_manifest.json"
    )
    _expect_sha(sfwd_pass_raw, component_hashes["sfwd_pass"], "copied SFWD PASS")
    _expect_sha(
        sfwd_manifest_raw,
        component_hashes["sfwd_manifest"],
        "copied SFWD manifest",
    )
    taw_pass_raw = _regular(
        arm / "logs/fr13_fixed32_taw_native_precompute.production_pass.json"
    )
    _expect_sha(taw_pass_raw, component_hashes["taw_production"], "copied TAW PASS")
    if _regular(
        arm / "logs/fr13_fixed32_taw_native_precompute_production.arm"
    ) != b"1\n":
        raise GateError("TAW production arm marker drifted")
    cfwd_pass_raw = _regular(arm / "logs/fr13_cfwd_logit_direct.production_pass.json")
    _expect_sha(
        cfwd_pass_raw,
        component_hashes["cfwd_credential"],
        "copied CFWD credential",
    )
    cfwd_payload, cfwd_raw = _load_json(Path(args.cfwd_credential).resolve())
    cfwd_gate._validate_credential(
        cfwd_payload,
        expected_source_commit=source_commit,
        expected_subset_sha256=cfwd_gate.GATE_SUBSET_SHA256,
    )
    _expect_sha(cfwd_raw, component_hashes["cfwd_credential"], "CFWD credential")
    cfwd_engagement, cfwd_engagement_raw = _load_json(
        arm / "logs/fr13_cfwd_logit_direct.production_engagement.json"
    )
    _validate_cfwd_engagement(
        cfwd_engagement,
        source_commit=source_commit,
        credential_sha256=component_hashes["cfwd_credential"],
    )

    qrow_sidecar_raw = _regular(
        arm / "logs/fr13_fa2_qrow32_b1_production_pass.json"
    )
    qrow_engagement, qrow_engagement_raw = _load_json(
        arm / "logs/fr13_fa2_qrow32_b1_production_engagement.json"
    )
    if (
        qrow_engagement.get("schema")
        != "fr13.fixed32.fa2_qrow32_b1_production_engagement.v2"
        or qrow_engagement.get("status") != "ENGAGED"
        or qrow_engagement.get("runtime_mode") != "FULL"
        or qrow_engagement.get("batch_size") != 1
        or qrow_engagement.get("arm") != "split2"
        or qrow_engagement.get("candidate_served") is not True
        or qrow_engagement.get("fallback_allowed") is not False
        or qrow_engagement.get("source_commit") != source_commit
        or qrow_engagement.get("pass_sidecar_sha256") != _sha256(qrow_sidecar_raw)
    ):
        raise GateError("Qrow32 split2 production engagement drifted")
    target_sidecar_raw = _regular(
        arm / "logs/fr13_fixed32_cutlass_streamk.production_pass.json"
    )
    target_binary, target_binary_raw = _load_json(
        arm / "logs/fr13_fixed32_cutlass_streamk_binary.json"
    )
    if (
        target_binary.get("schema") != "fr13.fixed32.cutlass_streamk_binary.v2"
        or target_binary.get("selector") != TARGET_SELECTOR
        or target_binary.get("production_enabled") is not True
        or target_binary.get("qualification_profile") != "k64_root"
        or not isinstance(target_binary.get("source"), dict)
        or target_binary["source"].get("sha256") != TARGET_SHA256
        or not isinstance(target_binary.get("destination"), dict)
        or target_binary["destination"].get("sha256") != TARGET_SHA256
        or target_binary.get("installed_mode") != "0555"
    ):
        raise GateError("target production binary evidence drifted")
    final_flush, final_flush_raw = _load_json(arm / "fixed32_final_flush.json")
    ack = final_flush.get("ack")
    if (
        final_flush.get("schema") != "fr13-fixed32-flush-client-result-v1"
        or not isinstance(ack, dict)
        or ack.get("schema") != "fr13-fixed32-flush-ack-v1"
        or ack.get("action") != "final"
        or ack.get("status") != "ok"
        or ack.get("mode") != "hydra27_fixed32"
    ):
        raise GateError("production smoke final flush drifted")

    runtime_hashes = {
        "runtime_manifest": _sha256(runtime_end),
        "external_manifest": _sha256(external_end),
        "health": _sha256(health_raw),
        "traffic_audit": _sha256(traffic_raw),
        "container_env": _sha256(container_raw),
        "docker_log": _sha256(docker_raw),
        "final_flush": _sha256(final_flush_raw),
        "qrow_sidecar": _sha256(qrow_sidecar_raw),
        "qrow_engagement": _sha256(qrow_engagement_raw),
        "target_sidecar": _sha256(target_sidecar_raw),
        "target_binary": _sha256(target_binary_raw),
        "taw_production_pass": _sha256(taw_pass_raw),
        "cfwd_production_pass": _sha256(cfwd_pass_raw),
        "cfwd_engagement": _sha256(cfwd_engagement_raw),
    }
    credential = {
        "schema": PRODUCTION_SMOKE_SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "mode": "hydra27_fixed32",
        "batch_size": 1,
        "task_id": TASK_ID,
        "task_count": 1,
        "subset_sha256": SUBSET_SHA256,
        "runtime_mode": "FULL",
        "production_paths": {
            "qrow32_split2": True,
            "gdn_gqa_group3": True,
            "dfwd_k64_top3": True,
            "target_gemm_selector": TARGET_SELECTOR,
            "sfwd_conv_postprep": True,
            "taw_native_precompute": True,
            "cfwd_logit_direct": True,
        },
        "component_credential_sha256s": component_hashes,
        "runtime_evidence_sha256s": runtime_hashes,
        "cfwd_served_return": CFWD_SERVED_RETURN,
        "performance_measurement": False,
        "timing_eligible": False,
        "exact4_eligible": True,
    }
    _validate_production_smoke_payload(
        credential,
        source_commit=source_commit,
        component_hashes=component_hashes,
    )
    _write_json(Path(args.output).resolve(), credential)


def validate_production_smoke(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    source_commit = str(args.source_commit)
    _validate_source_commit(repo, source_commit)
    component_hashes = {
        key: str(getattr(args, sha_name))
        for key, _path_name, sha_name in COMPONENT_ARGUMENTS
    }
    for key, value in component_hashes.items():
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise GateError(f"{key} SHA-256 is malformed")
    payload, raw = _load_json(Path(args.credential).resolve())
    _expect_sha(raw, args.expected_sha256, "production smoke credential")
    validated = _validate_production_smoke_payload(
        payload,
        source_commit=source_commit,
        component_hashes=component_hashes,
    )
    print(json.dumps(validated, ensure_ascii=True, sort_keys=True))


def issue_graph_gate(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    arm = Path(args.arm).resolve()
    source_commit = str(args.source_commit)
    _attestation, attestation_raw, candidate_raw = _validate_build(
        repo=repo,
        source_commit=source_commit,
        candidate_so=Path(args.candidate_so).resolve(),
        build_attestation=Path(args.build_attestation).resolve(),
    )
    runtime_launch = _regular(Path(args.runtime_launch).resolve())
    runtime_end = _regular(Path(args.runtime_end).resolve())
    external_launch_raw = _regular(Path(args.external_launch).resolve())
    external_end_raw = _regular(Path(args.external_end).resolve())
    if runtime_launch != runtime_end or external_launch_raw != external_end_raw:
        raise GateError("runtime or external manifest changed during graph gate")
    runtime, _ = _load_json(Path(args.runtime_end).resolve())
    try:
        regenerated_runtime = runtime_manifest.build_manifest(
            repo,
            profile="fixed32",
            sequence="scripts/fr13_fixed32_floor_timers_seq.sh",
            source_commit=source_commit,
        )
    except runtime_manifest.ManifestError as error:
        raise GateError(f"cannot regenerate runtime manifest: {error}") from error
    if runtime != regenerated_runtime:
        raise GateError("runtime manifest is not the canonical final-commit closure")
    if _regular(arm / "git_head.txt") != f"{source_commit}\n".encode("ascii"):
        raise GateError("arm git_head.txt is not bound to the final source commit")
    launcher_meta = _regular(arm.parent / "launcher_meta.txt").decode("ascii")
    if launcher_meta.splitlines().count(f"source_commit={source_commit}") != 1:
        raise GateError("launcher metadata is not bound to the final source commit")
    external, _ = _load_json(Path(args.external_end).resolve())
    try:
        fixed32_contract.validate_external_manifest(external)
    except fixed32_contract.ContractError as error:
        raise GateError(f"external manifest is invalid: {error}") from error

    qrow_arm = str(args.qrow_arm)
    qrow_live, qrow_raw = _load_json(Path(args.qrow_live).resolve())
    patch_sha = _sha256(_regular(repo / "scripts/fr13_patch_fa2_tree_bias.py"))
    qrow_summary = qrow32.validate_live_result(
        qrow_live,
        candidate_sha256=qrow32.CANDIDATE_SHA256,
        arm=qrow_arm,
        source_commit=source_commit,
        patch_source_sha256=patch_sha,
    )
    qrow32.validate_candidate(
        Path(args.qrow_candidate_so).resolve(), qrow32.CANDIDATE_SHA256
    )

    gdn, gdn_raw = _load_json(Path(args.gdn_credential).resolve())
    if (
        gdn.get("status") != "PASS"
        or gdn.get("candidate") != GQA3_CANDIDATE
        or gdn.get("source_commit") != source_commit
        or gdn.get("mode") != "hydra27_fixed32"
        or gdn.get("batch_size") != 1
        or gdn.get("concurrency") != 1
        or gdn.get("task_ids") != [TASK_ID]
        or gdn.get("raw_byte_equal") is not True
        or gdn.get("reference_served") is not True
    ):
        raise GateError("GQA3 credential is not the sibling graph-gate PASS")

    health_raw, traffic_raw = _validate_health_and_traffic(arm)
    container_env_raw = _regular(arm / "container_env.txt")
    _require_env(
        container_env_raw,
        (
            "FR13_FIXED32_MODE=hydra27_fixed32",
            "FR13_FIXED32_B1_DIAGNOSTIC=1",
            "FR13_DRAFT_VOCAB_ROOT=1",
            "FR13_DRAFT_VOCAB_K=65536",
            "MAX_NUM_SEQS=1",
            "SWE_CONCURRENCY=1",
            "ENFORCE_EAGER=0",
            "CUDAGRAPH_MODE=FULL_AND_PIECEWISE",
            f"FR13_FA2_QROW32_B1_LIVE_AB_ARM={qrow_arm}",
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE=gqa_group3",
            "FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH=1",
            "FR13_DFWD_K64_TOP3=1",
            f"FR13_DFWD_K64_TOP3_SHA256={DFWD_CANDIDATE_SHA256}",
        ),
    )
    docker_raw = _regular(arm / "docker_after_tasks.log")
    docker_text = docker_raw.decode("utf-8", errors="replace")
    markers = (
        "[FR13_DFWD_K64_TOP3] ready B1 K64 mapped width3",
        "[FR13_DFWD_K64_TOP3] engaged stock_argmax_topk_map_copy=0",
        "[FR13_DFWD_K64_TOP3] graph captured_calls=4",
    )
    if any(marker not in docker_text for marker in markers):
        raise GateError("DFWD K64 top3 ready/engaged/captured evidence is incomplete")

    common = {
        "source_commit": source_commit,
        "task_id": TASK_ID,
        "task_count": 1,
        "fixed32_mode": "hydra27_fixed32",
        "batch_size": 1,
        "concurrency": 1,
        "physical_rows": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "draft_vocab_blocks_sha256": BLOCK_MAP_SHA256,
        "runtime_manifest_sha256": _sha256(runtime_end),
        "external_manifest_sha256": _sha256(external_end_raw),
        "health_sha256": _sha256(health_raw),
        "traffic_audit_sha256": _sha256(traffic_raw),
        "container_env_sha256": _sha256(container_env_raw),
        "docker_log_sha256": _sha256(docker_raw),
        "gqa3_credential_sha256": _sha256(gdn_raw),
        "task_completion_verified": True,
        "authenticated_real_task_verified": True,
        "performance_measurement": False,
        "floor_acceptance_eligible": False,
        "production_enabled": False,
    }
    qrow_credential = {
        **common,
        "schema": (
            f"fr13.fixed32.fa2_qrow32_{qrow_arm}_k64_b1_live_verification.v2"
        ),
        "status": "PASS",
        "arm": qrow_arm,
        "selector_sentinel": qrow32.LIVE_ARMS[qrow_arm]["selector_sentinel"],
        "candidate_num_splits": qrow32.LIVE_ARMS[qrow_arm]["num_splits"],
        "candidate_so_sha256": qrow32.CANDIDATE_SHA256,
        "candidate_so_size": qrow32.CANDIDATE_SIZE,
        "fa2_head": qrow32.FA2_HEAD,
        "fa2_source_closure_sha256": qrow32.SOURCE_CLOSURE_SHA256,
        "patch_source_sha256": patch_sha,
        "live_result_sha256": _sha256(qrow_raw),
        "layers_sha256": qrow_summary["layers_sha256"],
        "reference_served": True,
        "raw_byte_equal": True,
    }
    dfwd_credential = {
        **common,
        "schema": DFWD_SCHEMA,
        "status": "PASS",
        "candidate_so_sha256": _sha256(candidate_raw),
        "candidate_so_bytes": len(candidate_raw),
        "candidate_source": DFWD_SOURCE,
        "candidate_source_sha256": DFWD_SOURCE_SHA256,
        "build_attestation_sha256": _sha256(attestation_raw),
        "ready_marker_verified": True,
        "engaged_marker_verified": True,
        "full_graph_capture_marker_verified": True,
        "minimum_reduction_launches_eliminated_per_event": 45,
        "drafter_byte_equality_claim": False,
        "drafter_proposal_semantics_changed": True,
        "target_verifier_and_rejection_sampling_remain_authoritative": True,
        "lossless_claim_scope": "unchanged verifier contract plus resolved real task",
    }
    _write_json(Path(args.qrow_output).resolve(), qrow_credential)
    _write_json(Path(args.dfwd_output).resolve(), dfwd_credential)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("validate-dfwd-build")
    build.add_argument("--repo", required=True)
    build.add_argument("--source-commit", required=True)
    build.add_argument("--candidate-so", required=True)
    build.add_argument("--build-attestation", required=True)
    build.set_defaults(function=validate_dfwd_build)
    credentials = commands.add_parser("validate-graph-credentials")
    credentials.add_argument("--repo", required=True)
    credentials.add_argument("--source-commit", required=True)
    credentials.add_argument("--qrow-live", required=True)
    credentials.add_argument("--qrow-live-sha256", required=True)
    credentials.add_argument("--qrow-credential", required=True)
    credentials.add_argument("--qrow-credential-sha256", required=True)
    credentials.add_argument("--gdn-credential", required=True)
    credentials.add_argument("--gdn-credential-sha256", required=True)
    credentials.add_argument("--dfwd-credential", required=True)
    credentials.add_argument("--dfwd-credential-sha256", required=True)
    credentials.add_argument("--candidate-so", required=True)
    credentials.add_argument("--build-attestation", required=True)
    credentials.set_defaults(function=validate_graph_credentials)
    eager = commands.add_parser("validate-eager-credentials")
    eager.add_argument("--repo", required=True)
    eager.add_argument("--source-commit", required=True)
    eager.add_argument("--combined-summary", required=True)
    eager.add_argument("--combined-summary-sha256", required=True)
    eager.add_argument("--target-live", required=True)
    eager.add_argument("--target-live-sha256", required=True)
    eager.add_argument("--sfwd-pass", required=True)
    eager.add_argument("--sfwd-pass-sha256", required=True)
    eager.add_argument("--source-manifest", required=True)
    eager.add_argument("--source-manifest-sha256", required=True)
    eager.set_defaults(function=validate_eager_credentials)
    issue = commands.add_parser("issue-graph-gate")
    issue.add_argument("--repo", required=True)
    issue.add_argument("--source-commit", required=True)
    issue.add_argument("--arm", required=True)
    issue.add_argument(
        "--qrow-arm", choices=tuple(qrow32.LIVE_ARMS), default=qrow32.ARM
    )
    issue.add_argument("--qrow-live", required=True)
    issue.add_argument("--qrow-candidate-so", required=True)
    issue.add_argument("--candidate-so", required=True)
    issue.add_argument("--build-attestation", required=True)
    issue.add_argument("--gdn-credential", required=True)
    issue.add_argument("--runtime-launch", required=True)
    issue.add_argument("--runtime-end", required=True)
    issue.add_argument("--external-launch", required=True)
    issue.add_argument("--external-end", required=True)
    issue.add_argument("--qrow-output", required=True)
    issue.add_argument("--dfwd-output", required=True)
    issue.set_defaults(function=issue_graph_gate)
    smoke = commands.add_parser("issue-production-smoke")
    smoke.add_argument("--repo", required=True)
    smoke.add_argument("--source-commit", required=True)
    smoke.add_argument("--arm", required=True)
    smoke.add_argument("--runtime-launch", required=True)
    smoke.add_argument("--runtime-end", required=True)
    smoke.add_argument("--external-launch", required=True)
    smoke.add_argument("--external-end", required=True)
    smoke.add_argument("--output", required=True)
    smoke_validate = commands.add_parser("validate-production-smoke")
    smoke_validate.add_argument("--repo", required=True)
    smoke_validate.add_argument("--source-commit", required=True)
    smoke_validate.add_argument("--credential", required=True)
    smoke_validate.add_argument("--expected-sha256", required=True)
    for _key, path_name, sha_name in COMPONENT_ARGUMENTS:
        smoke.add_argument("--" + path_name.replace("_", "-"), required=True)
        smoke.add_argument("--" + sha_name.replace("_", "-"), required=True)
        smoke_validate.add_argument(
            "--" + sha_name.replace("_", "-"), required=True
        )
    smoke.set_defaults(function=issue_production_smoke)
    smoke_validate.set_defaults(function=validate_production_smoke)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        args.function(args)
    except (GateError, ValueError) as error:
        raise SystemExit(str(error)) from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
