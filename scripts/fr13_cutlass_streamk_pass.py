#!/usr/bin/env python3
"""Issue and verify the fixed32 CUTLASS Stream-K production credential."""

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

import fr13_cutlass_wave_binary as binary
import fr13_hardware_floor_ledger as floor


LIVE_SCHEMA = "fr13.fixed32.cutlass_streamk_live_gate.v3"
SIDECAR_SCHEMA = "fr13.fixed32.cutlass_streamk.production_pass.v2"
K64_ROOT_LIVE_SCHEMA = "fr13.fixed32.cutlass_streamk_wide256_k64_root_live_gate.v1"
STATIC_PERSISTENT_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_static_persistent_k64_root_live_gate.v1"
)
DIVISOR_STATIC_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_divisor_static_k64_root_live_gate.v1"
)
IDENTITY_STAGE2_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stage2_k64_root_live_gate.v1"
)
IDENTITY_STAGE2_PINGPONG_B1_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stage2_pingpong_b1_k64_root_live_gate.v1"
)
IDENTITY_ONEN_B1_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_onen_b1_k64_root_live_gate.v1"
)
IDENTITY_ONEN_N5120_SINGLE_B1_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_onen_n5120_single_b1_k64_root_live_gate.v1"
)
IDENTITY_ONEN_N5120_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_onen_n5120_fullgrid_b1_k64_root_live_gate.v1"
)
IDENTITY_WIDE256_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_wide256_fullgrid_b1_k64_root_live_gate.v1"
)
K64_ROOT_SIDECAR_SCHEMA = "fr13.fixed32.cutlass_streamk.k64_root.production_pass.v1"
ATTESTATION_SCHEMA = "fr13.fixed32.cutlass_streamk_binary.v2"
PATCH_SOURCE = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
PATCH_SOURCE_SHA256 = "4132cd07388e0af0a3bc15c328eed74d08734e0bd6517b4d7418b72788b7e436"
DRAFT_VOCAB_BLOCKS_SOURCE = Path("scripts/fr13_dvk_subset_blocks.json")
DRAFT_VOCAB_BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
PATCHED_DISPATCH_SHA256 = (
    "f93500dc1ec4d19b93c13a8fec3a31e2fead23161ecb1c6839972697f6df25a4"
)
SOURCE_CONTRACTS = {
    "identity_onen_n5120_single_b1": {
        "patch_source_sha256": (
            "eadff808ef7db8de342d8c51e046cda9cc78bc4e308d1c1d08d5b33f7af1d2b0"
        ),
        "patched_dispatch_sha256": (
            "5e856f587480d2d04d9127b25e12d40ef82b8d07a2301389ab757523ce206d2d"
        ),
    },
    "identity_onen_n5120_fullgrid_b1": {
        "patch_source_sha256": (
            "623582b257a13f7551c81aaf8e87f7542ddb4d6564636f5e177ec0807126a341"
        ),
        "patched_dispatch_sha256": (
            "710da7d3a8e24c83f9f095222d5297d96f610c6310f3a8537ed1b925a25ece56"
        ),
    },
    "identity_wide256_fullgrid_b1": {
        "patch_source_sha256": (
            "ae9591a0c255c54bd8b5fed8576105013fce7f5f0834dbfb51ca1d455441f976"
        ),
        "patched_dispatch_sha256": (
            "569aea20321ba5461c4d3c9187aadf5390be363485f9aee538a738ef269ca6f0"
        ),
    },
}
SOURCE_BINDING_SCHEMA = "fr13.fixed32.cutlass_streamk.source_binding.v1"
RUNTIME_SOURCE_BINDING_SCHEMA = (
    "fr13.fixed32.cutlass_streamk.runtime_source_binding.v1"
)
HISTORICAL_QUALIFICATION_SELECTOR = "identity_wide256_fullgrid_b1"
HISTORICAL_QUALIFICATION_SOURCE_COMMIT = (
    "a8a904ed6c27a6338d43151038c155ebb76e3656"
)
HISTORICAL_QUALIFICATION_SOURCE_IDENTITY_SHA256 = (
    "fc062c7288770c81beca2660923a08cb136930400cee98fe64b87ce2a1c134ec"
)
HISTORICAL_QUALIFICATION_MODE = "historical_pinned"
HISTORICAL_QUALIFICATION_FIELDS = frozenset(
    {
        "qualification_source_mode",
        "runtime_source_commit",
        "runtime_source_identity",
    }
)
SOURCE_BINDING_PATHS = (
    "scripts/fr13_patch_cutlass_fixed32_wave.py",
    "scripts/fr13_run_b1_cutlass_streamk_live_gate.sh",
    "scripts/fr13_cutlass_streamk_pass.py",
    "scripts/fr13_cutlass_wave_binary.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_run_b1_kernel_live_gate.sh",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/run_swe_bench_q36_a.py",
    "scripts/fr13_fixed32_contract.py",
)
RUNTIME_SOURCE_BINDING_PATHS = SOURCE_BINDING_PATHS + (
    "scripts/fr13_runtime_manifest.py",
    "scripts/fr13_run_b1_target_sfwd_exact4_timing.sh",
)
WIDE256_LIVE_SCHEMA = "fr13.fixed32.cutlass_streamk_wide256_live_gate.v1"
DEFAULT_DIAGNOSTIC_TASK_PROFILE = "astropy12907"
DIAGNOSTIC_TASK_PROFILES = {
    DEFAULT_DIAGNOSTIC_TASK_PROFILE: {
        "task_ids": ("astropy__astropy-12907",),
        "task_marker": "swe_verified:astropy__astropy-12907",
        "candidate_selectors": None,
    },
    "astropy13236": {
        "task_ids": ("astropy__astropy-13236",),
        "task_marker": "swe_verified:astropy__astropy-13236",
        "candidate_selectors": frozenset(
            {
                "identity_onen_n5120_single_b1",
                "identity_onen_n5120_fullgrid_b1",
            }
        ),
    },
}
# Backward-compatible aliases for the original/default credential contract.
EXPECTED_TASK_IDS = DIAGNOSTIC_TASK_PROFILES[DEFAULT_DIAGNOSTIC_TASK_PROFILE][
    "task_ids"
]
EXPECTED_TASK_MARKER = DIAGNOSTIC_TASK_PROFILES[DEFAULT_DIAGNOSTIC_TASK_PROFILE][
    "task_marker"
]
EXPECTED_DRAFT_VOCAB_ROOT = 0
EXPECTED_DRAFT_VOCAB_K = 0
MAX_COMPARISONS = 320
EXPECTED_PROJECTION_NK = (
    (5120, 6144),
    (5120, 17408),
    (14336, 5120),
    (16384, 5120),
    (34816, 5120),
)
CANDIDATE_CONTRACTS = {
    "streamk_coop128": {
        "live_schema": LIVE_SCHEMA,
        "diagnostic_selector": "streamk_coop128_byte_ab",
    },
    "streamk_force_wide256": {
        "live_schema": WIDE256_LIVE_SCHEMA,
        "k64_root_live_schema": K64_ROOT_LIVE_SCHEMA,
        "diagnostic_selector": "streamk_force_wide256_byte_ab",
    },
    "static_persistent_stocktile": {
        "live_schema": "fr13.fixed32.cutlass_static_persistent_live_gate.v1",
        "k64_root_live_schema": STATIC_PERSISTENT_K64_ROOT_LIVE_SCHEMA,
        "diagnostic_selector": "static_persistent_stocktile_byte_ab",
    },
    "divisor_static_stocktile": {
        "live_schema": "fr13.fixed32.cutlass_divisor_static_live_gate.v1",
        "k64_root_live_schema": DIVISOR_STATIC_K64_ROOT_LIVE_SCHEMA,
        "diagnostic_selector": "divisor_static_stocktile_byte_ab",
    },
    "identity_stage2_static": {
        "live_schema": "fr13.fixed32.cutlass_identity_stage2_live_gate.v1",
        "k64_root_live_schema": IDENTITY_STAGE2_K64_ROOT_LIVE_SCHEMA,
        "diagnostic_selector": "identity_stage2_static_byte_ab",
    },
    "identity_stage2_pingpong_b1": {
        "live_schema": (
            "fr13.fixed32.cutlass_identity_stage2_pingpong_b1_live_gate.v1"
        ),
        "k64_root_live_schema": (
            IDENTITY_STAGE2_PINGPONG_B1_K64_ROOT_LIVE_SCHEMA
        ),
        "diagnostic_selector": "identity_stage2_pingpong_b1_byte_ab",
    },
    "identity_onen_b1": {
        "live_schema": "fr13.fixed32.cutlass_identity_onen_b1_live_gate.v1",
        "k64_root_live_schema": IDENTITY_ONEN_B1_K64_ROOT_LIVE_SCHEMA,
        "diagnostic_selector": "identity_onen_b1_byte_ab",
        "required_qualification_profile": "k64_root",
        "source_binding": "required",
    },
    "identity_onen_n5120_single_b1": {
        "live_schema": (
            "fr13.fixed32.cutlass_identity_onen_n5120_single_b1_live_gate.v1"
        ),
        "k64_root_live_schema": (
            IDENTITY_ONEN_N5120_SINGLE_B1_K64_ROOT_LIVE_SCHEMA
        ),
        "diagnostic_selector": "identity_onen_n5120_single_b1_byte_ab",
        "required_qualification_profile": "k64_root",
        "source_binding": "required",
    },
    "identity_onen_n5120_fullgrid_b1": {
        "live_schema": (
            "fr13.fixed32.cutlass_identity_onen_n5120_fullgrid_b1_live_gate.v1"
        ),
        "k64_root_live_schema": (
            IDENTITY_ONEN_N5120_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA
        ),
        "diagnostic_selector": "identity_onen_n5120_fullgrid_b1_byte_ab",
        "required_qualification_profile": "k64_root",
        "source_binding": "required",
    },
    "identity_wide256_fullgrid_b1": {
        "live_schema": (
            "fr13.fixed32.cutlass_identity_wide256_fullgrid_b1_live_gate.v1"
        ),
        "k64_root_live_schema": (
            IDENTITY_WIDE256_FULLGRID_B1_K64_ROOT_LIVE_SCHEMA
        ),
        "diagnostic_selector": "identity_wide256_fullgrid_b1_byte_ab",
        "required_qualification_profile": "k64_root",
        "source_binding": "required",
    },
}


def _diagnostic_task_profile(
    candidate_selector: str, diagnostic_task_profile: str
) -> dict[str, object]:
    try:
        profile = DIAGNOSTIC_TASK_PROFILES[diagnostic_task_profile]
    except KeyError as error:
        raise QualificationError(
            f"unsupported diagnostic task profile: {diagnostic_task_profile!r}"
        ) from error
    allowed = profile["candidate_selectors"]
    if allowed is not None and candidate_selector not in allowed:
        raise QualificationError(
            f"diagnostic task profile {diagnostic_task_profile!r} is not allowed "
            f"for {candidate_selector!r}"
        )
    return profile


def _validate_task_profile_binding(
    payload: dict[str, Any],
    key: str,
    diagnostic_task_profile: str,
    label: str,
) -> None:
    actual = payload.get(key)
    if actual == diagnostic_task_profile:
        return
    if (
        actual is None
        and diagnostic_task_profile == DEFAULT_DIAGNOSTIC_TASK_PROFILE
    ):
        return
    raise QualificationError(
        f"{label} {key} mismatch: {actual!r} != {diagnostic_task_profile!r}"
    )


QUALIFICATION_PROFILES: dict[str, dict[str, object]] = {
    "full_vocab": {
        "live_schema": None,
        "sidecar_schema": SIDECAR_SCHEMA,
        "binding_schema": "fr13.fixed32.cutlass_streamk.production_binding.v1",
        "run_classification": "one_real_swe_verified_b1_byte_diagnostic",
        "draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
        "requires_block_map": False,
    },
    "k64_root": {
        "live_schema": K64_ROOT_LIVE_SCHEMA,
        "sidecar_schema": K64_ROOT_SIDECAR_SCHEMA,
        "binding_schema": (
            "fr13.fixed32.cutlass_streamk.k64_root.production_binding.v1"
        ),
        "run_classification": (
            "one_real_swe_verified_b1_k64_root_byte_diagnostic"
        ),
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "mandatory_weight_bytes": floor.FIXED32_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": 137.6067177261,
        "requires_block_map": True,
    },
}


def _candidate_contract(candidate_selector: str) -> dict[str, object]:
    try:
        return CANDIDATE_CONTRACTS[candidate_selector]
    except KeyError as error:
        raise QualificationError(
            f"Stream-K candidate selector mismatch: {candidate_selector!r}"
        ) from error


def _source_contract(candidate_selector: str) -> dict[str, str]:
    try:
        return SOURCE_CONTRACTS[candidate_selector]
    except KeyError:
        return {
            "patch_source_sha256": PATCH_SOURCE_SHA256,
            "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        }


def _qualification_profile(
    candidate_selector: str, name: str
) -> dict[str, object]:
    contract = _candidate_contract(candidate_selector)
    required_profile = contract.get("required_qualification_profile")
    if required_profile is not None and name != required_profile:
        raise QualificationError(
            f"{candidate_selector} qualification requires the "
            f"{required_profile} profile"
        )
    try:
        profile = QUALIFICATION_PROFILES[name]
    except KeyError as error:
        raise QualificationError(
            f"unsupported Stream-K qualification profile: {name!r}"
        ) from error
    if name == "k64_root" and candidate_selector not in {
        "streamk_force_wide256",
        "static_persistent_stocktile",
        "divisor_static_stocktile",
        "identity_stage2_static",
        "identity_stage2_pingpong_b1",
        "identity_onen_b1",
        "identity_onen_n5120_single_b1",
        "identity_onen_n5120_fullgrid_b1",
        "identity_wide256_fullgrid_b1",
    }:
        raise QualificationError(
            "B1 k64_root qualification is restricted to wide256 or "
            "static-persistent stock-tile candidates"
        )
    result = dict(profile)
    if name == "full_vocab":
        result["live_schema"] = contract["live_schema"]
    else:
        result["live_schema"] = contract["k64_root_live_schema"]
    return result


class QualificationError(ValueError):
    """The Stream-K qualification chain is incomplete or inconsistent."""


class _GitUnavailableError(QualificationError):
    """Git cannot be executed in the runtime verification environment."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise QualificationError(f"{label} does not exist: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise QualificationError(f"{label} is not a regular non-symlink file: {path}")
    return info


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise QualificationError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise QualificationError(f"non-finite JSON value: {value}")


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular_file(path, label)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationError(f"{label} is not canonical ASCII JSON") from error
    if not isinstance(payload, dict):
        raise QualificationError(f"{label} must contain a JSON object")
    return payload, raw


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QualificationError(f"{label} is not a lowercase SHA-256")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_raw = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_raw)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_patch_source(
    path: Path, candidate_selector: str
) -> dict[str, object]:
    info = _regular_file(path, "Stream-K patch source")
    digest = sha256_file(path)
    expected = _source_contract(candidate_selector)["patch_source_sha256"]
    if digest != expected:
        raise QualificationError(
            f"Stream-K patch source SHA-256 mismatch: {digest} != {expected}"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
    }


def _validate_draft_vocab_blocks(path: Path) -> dict[str, object]:
    info = _regular_file(path, "draft-vocabulary block map")
    digest = sha256_file(path)
    if digest != DRAFT_VOCAB_BLOCKS_SHA256:
        raise QualificationError(
            "draft-vocabulary block-map SHA-256 mismatch: "
            f"{digest} != {DRAFT_VOCAB_BLOCKS_SHA256}"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "container_path": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
    }


def _git_output(repo_root: Path, *arguments: str, label: str) -> bytes:
    try:
        process = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise _GitUnavailableError("git is unavailable for source binding") from error
    if process.returncode != 0:
        detail = process.stderr.decode("utf-8", errors="replace").strip()
        raise QualificationError(f"{label} failed: {detail or process.returncode}")
    return process.stdout


def _require_clean_tracked_tree(repo_root: Path) -> None:
    for cached in (False, True):
        arguments = ["diff"]
        if cached:
            arguments.append("--cached")
        arguments.extend(("--quiet", "--"))
        process = subprocess.run(
            ["git", "-C", os.fspath(repo_root), *arguments],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.returncode == 1:
            raise QualificationError(
                "source-binding repository has a dirty tracked working tree"
            )
        if process.returncode != 0:
            detail = process.stderr.decode("utf-8", errors="replace").strip()
            raise QualificationError(
                f"source-binding worktree check failed: "
                f"{detail or process.returncode}"
            )


def validate_source_commit_binding(
    source_commit: str,
    patch_source: Path = PATCH_SOURCE,
    candidate_selector: str = "identity_onen_b1",
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise QualificationError("source-binding commit is invalid")
    _regular_file(patch_source, "Stream-K patch source")
    patch_source = patch_source.resolve(strict=True)
    repo_raw = _git_output(
        patch_source.parent,
        "rev-parse",
        "--show-toplevel",
        label="source-binding repository discovery",
    )
    try:
        repo_root = Path(repo_raw.decode("utf-8").strip()).resolve(strict=True)
    except (UnicodeDecodeError, FileNotFoundError) as error:
        raise QualificationError("source-binding repository root is invalid") from error
    expected_patch_source = (repo_root / PATCH_SOURCE).resolve(strict=True)
    if patch_source != expected_patch_source:
        raise QualificationError(
            "source-binding patch source is not the canonical repository path"
        )
    resolved_commit = _git_output(
        repo_root,
        "rev-parse",
        "--verify",
        f"{source_commit}^{{commit}}",
        label="source-binding commit resolution",
    ).decode("ascii").strip()
    if resolved_commit != source_commit:
        raise QualificationError("source-binding commit does not resolve exactly")
    head_commit = _git_output(
        repo_root,
        "rev-parse",
        "HEAD",
        label="source-binding HEAD resolution",
    ).decode("ascii").strip()
    if head_commit != source_commit:
        raise QualificationError(
            f"source-binding runtime commit mismatch: {head_commit} != {source_commit}"
        )
    _require_clean_tracked_tree(repo_root)

    files: dict[str, object] = {}
    for relative in SOURCE_BINDING_PATHS:
        working_path = repo_root / relative
        info = _regular_file(working_path, f"source-binding file {relative}")
        committed = _git_output(
            repo_root,
            "show",
            f"{source_commit}:{relative}",
            label=f"source-binding git show for {relative}",
        )
        working = working_path.read_bytes()
        if working != committed:
            raise QualificationError(
                f"source-binding working bytes differ from commit for {relative}"
            )
        files[relative] = {
            "bytes": info.st_size,
            "sha256": hashlib.sha256(committed).hexdigest(),
        }
    patch_identity = files[os.fspath(PATCH_SOURCE)]
    assert isinstance(patch_identity, dict)
    expected_patch_sha256 = _source_contract(candidate_selector)[
        "patch_source_sha256"
    ]
    if patch_identity.get("sha256") != expected_patch_sha256:
        raise QualificationError(
            "source-binding committed patch source does not match the pinned digest"
        )
    return {
        "schema": SOURCE_BINDING_SCHEMA,
        "source_commit": source_commit,
        "files": files,
    }


def _canonical_identity_sha256(source_identity: object) -> str:
    try:
        encoded = json.dumps(
            source_identity,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, UnicodeError) as error:
        raise QualificationError("source identity is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _canonical_source_repository(patch_source: Path) -> tuple[Path, Path]:
    _regular_file(patch_source, "Stream-K patch source")
    resolved_patch_source = patch_source.resolve(strict=True)
    repo_root = resolved_patch_source.parents[len(PATCH_SOURCE.parts) - 1]
    if (repo_root / PATCH_SOURCE).resolve(strict=True) != resolved_patch_source:
        raise QualificationError(
            "source-binding patch source is not the canonical repository path"
        )
    return repo_root, resolved_patch_source


def validate_historical_qualification_source_binding(
    source_commit: str,
    source_identity: object,
    patch_source: Path,
    candidate_selector: str,
) -> dict[str, object]:
    if (
        candidate_selector != HISTORICAL_QUALIFICATION_SELECTOR
        or source_commit != HISTORICAL_QUALIFICATION_SOURCE_COMMIT
    ):
        raise QualificationError(
            "historical qualification is restricted to the pinned cooperative B1 target"
        )
    if (
        not isinstance(source_identity, dict)
        or _canonical_identity_sha256(source_identity)
        != HISTORICAL_QUALIFICATION_SOURCE_IDENTITY_SHA256
    ):
        raise QualificationError("historical qualification source identity mismatch")
    if source_identity.get("schema") != SOURCE_BINDING_SCHEMA:
        raise QualificationError("historical qualification source schema mismatch")
    if source_identity.get("source_commit") != source_commit:
        raise QualificationError("historical qualification source commit mismatch")
    records = source_identity.get("files")
    if not isinstance(records, dict) or set(records) != set(SOURCE_BINDING_PATHS):
        raise QualificationError("historical qualification source manifest mismatch")

    repo_root, _ = _canonical_source_repository(patch_source)
    resolved_commit = _git_output(
        repo_root,
        "rev-parse",
        "--verify",
        f"{source_commit}^{{commit}}",
        label="historical qualification commit resolution",
    ).decode("ascii").strip()
    if resolved_commit != source_commit:
        raise QualificationError(
            "historical qualification commit does not resolve exactly"
        )
    for relative in SOURCE_BINDING_PATHS:
        record = records.get(relative)
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise QualificationError(
                f"historical qualification source record is invalid for {relative}"
            )
        committed = _git_output(
            repo_root,
            "show",
            f"{source_commit}:{relative}",
            label=f"historical qualification git show for {relative}",
        )
        expected = {
            "bytes": len(committed),
            "sha256": hashlib.sha256(committed).hexdigest(),
        }
        if record != expected:
            raise QualificationError(
                f"historical qualification source record mismatch for {relative}"
            )
    patch_record = records[os.fspath(PATCH_SOURCE)]
    assert isinstance(patch_record, dict)
    if patch_record.get("sha256") != _source_contract(candidate_selector)[
        "patch_source_sha256"
    ]:
        raise QualificationError(
            "historical qualification patch source does not match the candidate"
        )
    return source_identity


def validate_runtime_source_commit_identity(
    runtime_source_commit: str,
    patch_source: Path = PATCH_SOURCE,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", runtime_source_commit) is None:
        raise QualificationError("runtime source-binding commit is invalid")
    repo_root, _ = _canonical_source_repository(patch_source)
    resolved_commit = _git_output(
        repo_root,
        "rev-parse",
        "--verify",
        f"{runtime_source_commit}^{{commit}}",
        label="runtime source-binding commit resolution",
    ).decode("ascii").strip()
    head_commit = _git_output(
        repo_root,
        "rev-parse",
        "HEAD",
        label="runtime source-binding HEAD resolution",
    ).decode("ascii").strip()
    if resolved_commit != runtime_source_commit or head_commit != runtime_source_commit:
        raise QualificationError(
            "runtime source-binding commit does not equal the current HEAD"
        )
    _require_clean_tracked_tree(repo_root)

    files: dict[str, object] = {}
    for relative in RUNTIME_SOURCE_BINDING_PATHS:
        working_path = repo_root / relative
        info = _regular_file(working_path, f"runtime source-binding file {relative}")
        committed = _git_output(
            repo_root,
            "show",
            f"{runtime_source_commit}:{relative}",
            label=f"runtime source-binding git show for {relative}",
        )
        working = working_path.read_bytes()
        if working != committed:
            raise QualificationError(
                f"runtime source-binding working bytes differ from commit for {relative}"
            )
        files[relative] = {
            "bytes": info.st_size,
            "sha256": hashlib.sha256(committed).hexdigest(),
        }
    return {
        "schema": RUNTIME_SOURCE_BINDING_SCHEMA,
        "source_commit": runtime_source_commit,
        "files": files,
    }


def _historical_qualification_requested(payload: dict[str, Any]) -> bool:
    present = HISTORICAL_QUALIFICATION_FIELDS.intersection(payload)
    if present and present != HISTORICAL_QUALIFICATION_FIELDS:
        raise QualificationError(
            "historical qualification runtime source fields are incomplete"
        )
    if not present:
        return False
    if payload.get("qualification_source_mode") != HISTORICAL_QUALIFICATION_MODE:
        raise QualificationError("historical qualification source mode mismatch")
    return True


def _validate_mounted_source_identity(
    source_commit: str,
    source_identity: object,
    patch_source: Path,
    candidate_selector: str,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise QualificationError("source-binding commit is invalid")
    if not isinstance(source_identity, dict) or set(source_identity) != {
        "schema",
        "source_commit",
        "files",
    }:
        raise QualificationError("mounted source identity has an invalid structure")
    if source_identity.get("schema") != SOURCE_BINDING_SCHEMA:
        raise QualificationError("mounted source identity schema mismatch")
    if source_identity.get("source_commit") != source_commit:
        raise QualificationError("mounted source identity commit mismatch")
    files = source_identity.get("files")
    if not isinstance(files, dict) or set(files) != set(SOURCE_BINDING_PATHS):
        raise QualificationError("mounted source identity file manifest mismatch")

    _regular_file(patch_source, "Stream-K patch source")
    resolved_patch_source = patch_source.resolve(strict=True)
    repo_root = resolved_patch_source.parents[len(PATCH_SOURCE.parts) - 1]
    if repo_root / PATCH_SOURCE != resolved_patch_source:
        raise QualificationError(
            "mounted patch source is not the canonical source-binding path"
        )

    for relative in SOURCE_BINDING_PATHS:
        record = files.get(relative)
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise QualificationError(
                f"mounted source identity record is invalid for {relative}"
            )
        expected_bytes = record.get("bytes")
        expected_sha256 = record.get("sha256")
        if (
            not isinstance(expected_bytes, int)
            or isinstance(expected_bytes, bool)
            or expected_bytes < 0
            or not isinstance(expected_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
        ):
            raise QualificationError(
                f"mounted source identity record is invalid for {relative}"
            )
        working_path = repo_root / relative
        info = _regular_file(working_path, f"mounted source-binding file {relative}")
        if info.st_size != expected_bytes:
            raise QualificationError(
                f"mounted source-binding file size mismatch for {relative}"
            )
        if sha256_file(working_path) != expected_sha256:
            raise QualificationError(
                f"mounted source-binding file SHA-256 mismatch for {relative}"
            )

    patch_record = files[os.fspath(PATCH_SOURCE)]
    assert isinstance(patch_record, dict)
    if (
        patch_record.get("sha256")
        != _source_contract(candidate_selector)["patch_source_sha256"]
    ):
        raise QualificationError(
            "mounted source-binding patch source does not match the pinned digest"
        )
    return source_identity


def _validate_runtime_source_commit_binding(
    source_commit: str,
    source_identity: object,
    patch_source: Path,
    candidate_selector: str,
) -> dict[str, object]:
    try:
        return validate_source_commit_binding(
            source_commit, patch_source, candidate_selector
        )
    except _GitUnavailableError:
        return _validate_mounted_source_identity(
            source_commit, source_identity, patch_source, candidate_selector
        )


def validate_live_result(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = "streamk_coop128",
    qualification_profile: str = "full_vocab",
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    diagnostic_task_profile: str = DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    runtime_source_commit: str | None = None,
) -> dict[str, Any]:
    candidate_contract = _candidate_contract(candidate_selector)
    profile = _qualification_profile(candidate_selector, qualification_profile)
    task_profile = _diagnostic_task_profile(
        candidate_selector, diagnostic_task_profile
    )
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected live-result SHA-256"
    )
    payload, raw = _read_json(live_result, "Stream-K live PASS")
    live_sha256 = hashlib.sha256(raw).hexdigest()
    if live_sha256 != expected_live_sha256:
        raise QualificationError(
            f"Stream-K live PASS SHA-256 mismatch: {live_sha256} != {expected_live_sha256}"
        )

    candidate = binary.verify_candidate(
        candidate_so,
        diagnostic_selector,
        qualification_profile=qualification_profile,
    )
    source_contract = _source_contract(candidate_selector)
    historical_qualification = runtime_source_commit is not None
    runtime_source_identity: dict[str, object] | None = None
    if historical_qualification:
        if candidate_selector != HISTORICAL_QUALIFICATION_SELECTOR:
            raise QualificationError(
                "runtime source separation is restricted to the historical "
                "cooperative B1 target"
            )
        patch = {"sha256": source_contract["patch_source_sha256"]}
    else:
        patch = _validate_patch_source(patch_source, candidate_selector)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    expected_fields: dict[str, object] = {
        "schema": profile["live_schema"],
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(task_profile["task_ids"]),
        "task_marker": task_profile["task_marker"],
        "draft_vocab_root": profile["draft_vocab_root"],
        "draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": candidate_selector,
        "diagnostic_selector": diagnostic_selector,
        "served_result": "stock",
        "production_enabled": False,
        "observed_m_values": [32],
        "observed_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": source_contract["patch_source_sha256"],
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": source_contract["patched_dispatch_sha256"],
        "errors": [],
    }
    if candidate_selector in {
        "streamk_force_wide256",
        "static_persistent_stocktile",
        "divisor_static_stocktile",
        "identity_wide256_fullgrid_b1",
    }:
        expected_fields["candidate_family"] = candidate["candidate_family"]
    if qualification_profile == "k64_root":
        assert block_map is not None
        expected_fields.update(
            {
                "qualification_profile": qualification_profile,
                "draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                "draft_vocab_blocks_sha256": block_map["sha256"],
                "comparison_call_limit": MAX_COMPARISONS,
            }
        )
    _validate_task_profile_binding(
        payload,
        "diagnostic_task_profile",
        diagnostic_task_profile,
        "Stream-K live PASS",
    )
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"Stream-K live PASS {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    comparisons = payload.get("comparisons")
    if (
        isinstance(comparisons, bool)
        or not isinstance(comparisons, int)
        or comparisons < len(EXPECTED_PROJECTION_NK)
        or comparisons > (
            MAX_COMPARISONS if qualification_profile == "k64_root" else 256
        )
    ):
        raise QualificationError("Stream-K live PASS comparison count is invalid")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("Stream-K live PASS source commit is invalid")
    if expected_source_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None:
            raise QualificationError("expected source commit is invalid")
        if source_commit != expected_source_commit:
            raise QualificationError(
                "Stream-K live PASS source commit is stale: "
                f"{source_commit} != {expected_source_commit}"
            )
    source_identity: dict[str, object] | None = None
    if candidate_contract.get("source_binding") == "required":
        if historical_qualification:
            source_identity = validate_historical_qualification_source_binding(
                source_commit,
                payload.get("source_identity"),
                patch_source,
                candidate_selector,
            )
            assert runtime_source_commit is not None
            if runtime_source_commit == source_commit:
                raise QualificationError(
                    "historical qualification and runtime source commits must differ"
                )
            runtime_source_identity = validate_runtime_source_commit_identity(
                runtime_source_commit, patch_source
            )
        else:
            source_identity = validate_source_commit_binding(
                source_commit, patch_source, candidate_selector
            )
        if payload.get("source_identity") != source_identity:
            raise QualificationError(
                "Stream-K live PASS source-identity binding mismatch"
            )
    attestation_sha256 = _require_sha256(
        payload.get("binary_attestation_sha256"), "binary attestation SHA-256"
    )
    real_task_arm_sha256 = _require_sha256(
        payload.get("real_task_arm_sha256"), "real-task arm SHA-256"
    )
    container_env_sha256 = _require_sha256(
        payload.get("container_env_sha256"), "container environment SHA-256"
    )
    result = {
        "schema": profile["sidecar_schema"],
        "status": "QUALIFIED",
        "candidate_selector": candidate_selector,
        "diagnostic_selector": diagnostic_selector,
        "candidate_family": candidate["candidate_family"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "live_result_sha256": live_sha256,
        "binary_attestation_sha256": attestation_sha256,
        "patch_source_sha256": patch["sha256"],
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": source_contract["patched_dispatch_sha256"],
        "qualification_source_commit": source_commit,
        "qualification_task_profile": diagnostic_task_profile,
        "qualification_task_ids": list(task_profile["task_ids"]),
        "qualification_task_marker": task_profile["task_marker"],
        "real_task_arm_sha256": real_task_arm_sha256,
        "container_env_sha256": container_env_sha256,
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 32,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
    if qualification_profile == "k64_root":
        assert block_map is not None
        result.update(
            {
                "qualification_profile": qualification_profile,
                "qualified_draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
                "qualified_comparison_call_limit": MAX_COMPARISONS,
            }
        )
    if source_identity is not None:
        result["qualification_source_identity"] = source_identity
    if runtime_source_identity is not None:
        assert runtime_source_commit is not None
        result.update(
            {
                "qualification_source_mode": HISTORICAL_QUALIFICATION_MODE,
                "runtime_source_commit": runtime_source_commit,
                "runtime_source_identity": runtime_source_identity,
            }
        )
    return result


def issue_sidecar(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    output: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = "streamk_coop128",
    qualification_profile: str = "full_vocab",
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    diagnostic_task_profile: str = DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    runtime_source_commit: str | None = None,
) -> dict[str, Any]:
    payload = validate_live_result(
        live_result,
        expected_live_sha256,
        candidate_so,
        patch_source,
        expected_source_commit,
        candidate_selector,
        qualification_profile,
        draft_vocab_blocks,
        diagnostic_task_profile,
        runtime_source_commit,
    )
    _write_json(output, payload)
    return payload


def verify_sidecar(
    sidecar: Path,
    expected_sidecar_sha256: str,
    candidate_so: Path,
    patch_source: Path = PATCH_SOURCE,
    *,
    candidate_selector: str | None = None,
    qualification_profile: str | None = None,
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    diagnostic_task_profile: str = DEFAULT_DIAGNOSTIC_TASK_PROFILE,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(sidecar, "Stream-K production sidecar")
    sidecar_selector = payload.get("candidate_selector")
    if candidate_selector is not None and sidecar_selector != candidate_selector:
        raise QualificationError("Stream-K production sidecar selector mismatch")
    if not isinstance(sidecar_selector, str):
        raise QualificationError("Stream-K production sidecar selector is invalid")
    candidate_contract = _candidate_contract(sidecar_selector)
    schema = payload.get("schema")
    if schema == SIDECAR_SCHEMA:
        sidecar_profile = "full_vocab"
    elif schema == K64_ROOT_SIDECAR_SCHEMA:
        sidecar_profile = "k64_root"
    else:
        raise QualificationError("Stream-K production sidecar schema mismatch")
    if qualification_profile is not None and sidecar_profile != qualification_profile:
        raise QualificationError(
            "Stream-K production sidecar qualification-profile mismatch"
        )
    profile = _qualification_profile(sidecar_selector, sidecar_profile)
    task_profile = _diagnostic_task_profile(
        sidecar_selector, diagnostic_task_profile
    )
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sidecar_sha256:
        raise QualificationError(
            "Stream-K production sidecar SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sidecar_sha256}"
        )
    candidate = binary.verify_candidate(
        candidate_so,
        diagnostic_selector,
        qualification_profile=sidecar_profile,
    )
    source_contract = _source_contract(sidecar_selector)
    historical_qualification = _historical_qualification_requested(payload)
    if historical_qualification:
        if sidecar_selector != HISTORICAL_QUALIFICATION_SELECTOR:
            raise QualificationError(
                "historical qualification sidecar selector mismatch"
            )
        patch = {"sha256": source_contract["patch_source_sha256"]}
    else:
        patch = _validate_patch_source(patch_source, sidecar_selector)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    required = {
        "schema": profile["sidecar_schema"],
        "status": "QUALIFIED",
        "candidate_selector": sidecar_selector,
        "diagnostic_selector": diagnostic_selector,
        "candidate_family": candidate["candidate_family"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": patch["sha256"],
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": source_contract["patched_dispatch_sha256"],
        "qualification_task_ids": list(task_profile["task_ids"]),
        "qualification_task_marker": task_profile["task_marker"],
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 32,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
    if sidecar_profile == "k64_root":
        assert block_map is not None
        required.update(
            {
                "qualification_profile": sidecar_profile,
                "qualified_draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
                "qualified_comparison_call_limit": MAX_COMPARISONS,
            }
        )
    _validate_task_profile_binding(
        payload,
        "qualification_task_profile",
        diagnostic_task_profile,
        "Stream-K production sidecar",
    )
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"Stream-K production sidecar {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    _require_sha256(payload.get("live_result_sha256"), "sidecar live-result SHA-256")
    _require_sha256(
        payload.get("binary_attestation_sha256"),
        "sidecar diagnostic binary-attestation SHA-256",
    )
    _require_sha256(
        payload.get("real_task_arm_sha256"),
        "sidecar real-task arm SHA-256",
    )
    _require_sha256(
        payload.get("container_env_sha256"),
        "sidecar container environment SHA-256",
    )
    source_commit = payload.get("qualification_source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("sidecar qualification source commit is invalid")
    if candidate_contract.get("source_binding") == "required":
        if historical_qualification:
            source_identity = validate_historical_qualification_source_binding(
                source_commit,
                payload.get("qualification_source_identity"),
                patch_source,
                sidecar_selector,
            )
            runtime_source_commit = payload.get("runtime_source_commit")
            if (
                not isinstance(runtime_source_commit, str)
                or runtime_source_commit == source_commit
            ):
                raise QualificationError(
                    "historical qualification runtime source commit is invalid"
                )
            runtime_source_identity = validate_runtime_source_commit_identity(
                runtime_source_commit, patch_source
            )
            if payload.get("runtime_source_identity") != runtime_source_identity:
                raise QualificationError(
                    "Stream-K production sidecar runtime source binding mismatch"
                )
        else:
            source_identity = _validate_runtime_source_commit_binding(
                source_commit,
                payload.get("qualification_source_identity"),
                patch_source,
                sidecar_selector,
            )
        if payload.get("qualification_source_identity") != source_identity:
            raise QualificationError(
                "Stream-K production sidecar source-identity binding mismatch"
            )
    return payload


def validate_production_attestation(
    attestation: Path,
    expected_sidecar_sha256: str,
    qualification_profile: str | None = None,
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    patch_source: Path = PATCH_SOURCE,
    diagnostic_task_profile: str = DEFAULT_DIAGNOSTIC_TASK_PROFILE,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(attestation, "Stream-K binary attestation")
    candidate_selector = payload.get("selector")
    if not isinstance(candidate_selector, str):
        raise QualificationError("Stream-K binary attestation selector is invalid")
    candidate_contract = _candidate_contract(candidate_selector)
    task_profile = _diagnostic_task_profile(
        candidate_selector, diagnostic_task_profile
    )
    source_contract = _source_contract(candidate_selector)
    candidate_sha256, candidate_bytes, candidate_family = binary.candidate_identity(
        candidate_selector
    )
    required = {
        "schema": ATTESTATION_SCHEMA,
        "selector": candidate_selector,
        "installed_mode": "0555",
        "production_enabled": True,
    }
    required_profile = candidate_contract.get("required_qualification_profile")
    if required_profile is not None:
        required["qualification_profile"] = required_profile
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"Stream-K binary attestation {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    for label, expected_path in (
        ("source", binary.CONTAINER_SOURCE),
        ("destination", binary.CONTAINER_DESTINATION),
    ):
        identity = payload.get(label)
        if not isinstance(identity, dict):
            raise QualificationError(f"Stream-K binary attestation lacks {label}")
        expected_identity = {
            "path": str(expected_path),
            "bytes": candidate_bytes,
            "sha256": candidate_sha256,
            "regular": True,
            "symlink": False,
        }
        for key, expected in expected_identity.items():
            if identity.get(key) != expected:
                raise QualificationError(
                    f"Stream-K binary attestation {label}.{key} mismatch"
                )
    qualification = payload.get("qualification")
    if not isinstance(qualification, dict):
        raise QualificationError("Stream-K binary attestation lacks qualification")
    historical_qualification = _historical_qualification_requested(qualification)
    if (
        historical_qualification
        and candidate_selector != HISTORICAL_QUALIFICATION_SELECTOR
    ):
        raise QualificationError("historical qualification attestation selector mismatch")
    _validate_task_profile_binding(
        qualification,
        "qualification_task_profile",
        diagnostic_task_profile,
        "Stream-K attestation qualification",
    )
    qualification_task_ids = qualification.get("qualification_task_ids")
    if not (
        qualification_task_ids == list(task_profile["task_ids"])
        or (
            qualification_task_ids is None
            and diagnostic_task_profile == DEFAULT_DIAGNOSTIC_TASK_PROFILE
        )
    ):
        raise QualificationError(
            "Stream-K attestation qualification_task_ids binding mismatch"
        )
    attested_profile = qualification.get("qualification_profile", "full_vocab")
    if not isinstance(attested_profile, str):
        raise QualificationError(
            "Stream-K attestation qualification profile is invalid"
        )
    if qualification_profile is not None and attested_profile != qualification_profile:
        raise QualificationError(
            "Stream-K attestation qualification-profile binding mismatch"
        )
    profile = _qualification_profile(candidate_selector, attested_profile)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    if qualification.get("sidecar_sha256") != expected_sidecar_sha256:
        raise QualificationError("Stream-K attestation sidecar binding mismatch")
    if qualification.get("candidate_sha256") != candidate_sha256:
        raise QualificationError("Stream-K attestation candidate binding mismatch")
    if qualification.get("patch_source_sha256") != source_contract[
        "patch_source_sha256"
    ]:
        raise QualificationError("Stream-K attestation patch-source binding mismatch")
    for key, expected in (
        ("qualification_task_marker", task_profile["task_marker"]),
        ("qualified_draft_vocab_root", profile["draft_vocab_root"]),
        ("qualified_draft_vocab_k", profile["draft_vocab_k"]),
        ("mandatory_weight_bytes", profile["mandatory_weight_bytes"]),
        (
            "mandatory_weight_floor_ms",
            profile["mandatory_weight_floor_ms"],
        ),
        ("one_sided_u95_cap_ms", profile["one_sided_u95_cap_ms"]),
    ):
        if qualification.get(key) != expected:
            raise QualificationError(
                f"Stream-K attestation {key} binding mismatch"
            )
    if attested_profile == "k64_root":
        assert block_map is not None
        for key, expected in (
            ("qualification_profile", attested_profile),
            (
                "qualified_draft_vocab_blocks",
                DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
            ),
            ("qualified_draft_vocab_blocks_sha256", block_map["sha256"]),
            ("qualified_comparison_call_limit", MAX_COMPARISONS),
            ("qualified_fixed_rows", 32),
            (
                "qualified_projection_nk",
                [list(shape) for shape in EXPECTED_PROJECTION_NK],
            ),
        ):
            if qualification.get(key) != expected:
                raise QualificationError(
                    f"Stream-K attestation {key} binding mismatch"
                )
    real_task_arm_sha256 = _require_sha256(
        qualification.get("real_task_arm_sha256"),
        "attestation real-task arm SHA-256",
    )
    container_env_sha256 = _require_sha256(
        qualification.get("container_env_sha256"),
        "attestation container environment SHA-256",
    )
    qualification_source_commit = qualification.get("qualification_source_commit")
    if (
        not isinstance(qualification_source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", qualification_source_commit) is None
    ):
        raise QualificationError(
            "Stream-K attestation qualification source commit is invalid"
        )
    source_identity: dict[str, object] | None = None
    runtime_source_identity: dict[str, object] | None = None
    runtime_source_commit: str | None = None
    if candidate_contract.get("source_binding") == "required":
        if historical_qualification:
            source_identity = validate_historical_qualification_source_binding(
                qualification_source_commit,
                qualification.get("qualification_source_identity"),
                patch_source,
                candidate_selector,
            )
            runtime_source_commit_raw = qualification.get("runtime_source_commit")
            if (
                not isinstance(runtime_source_commit_raw, str)
                or runtime_source_commit_raw == qualification_source_commit
            ):
                raise QualificationError(
                    "Stream-K attestation runtime source commit is invalid"
                )
            runtime_source_commit = runtime_source_commit_raw
            runtime_source_identity = validate_runtime_source_commit_identity(
                runtime_source_commit, patch_source
            )
            if qualification.get("runtime_source_identity") != runtime_source_identity:
                raise QualificationError(
                    "Stream-K attestation runtime source-identity binding mismatch"
                )
        else:
            source_identity = _validate_runtime_source_commit_binding(
                qualification_source_commit,
                qualification.get("qualification_source_identity"),
                patch_source,
                candidate_selector,
            )
        if qualification.get("qualification_source_identity") != source_identity:
            raise QualificationError(
                "Stream-K attestation source-identity binding mismatch"
            )
    _require_sha256(
        qualification.get("live_result_sha256"),
        "attestation live-result SHA-256",
    )
    result = {
        "schema": profile["binding_schema"],
        "status": "BOUND",
        "selector": candidate_selector,
        "diagnostic_selector": candidate_contract["diagnostic_selector"],
        "candidate_family": candidate_family,
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": candidate_bytes,
        "patch_source_sha256": source_contract["patch_source_sha256"],
        "production_sidecar_sha256": expected_sidecar_sha256,
        "live_result_sha256": qualification["live_result_sha256"],
        "binary_attestation_sha256": hashlib.sha256(raw).hexdigest(),
        "qualification_source_commit": qualification_source_commit,
        "qualification_task_profile": diagnostic_task_profile,
        "qualification_task_ids": list(task_profile["task_ids"]),
        "qualification_task_marker": task_profile["task_marker"],
        "real_task_arm_sha256": real_task_arm_sha256,
        "container_env_sha256": container_env_sha256,
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "installed_mode": "0555",
        "production_default_enabled": False,
    }
    if attested_profile == "k64_root":
        assert block_map is not None
        result.update(
            {
                "qualification_profile": attested_profile,
                "qualified_draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
                "qualified_comparison_call_limit": MAX_COMPARISONS,
            }
        )
    if source_identity is not None:
        result["qualification_source_identity"] = source_identity
    if runtime_source_identity is not None:
        assert runtime_source_commit is not None
        result.update(
            {
                "qualification_source_mode": HISTORICAL_QUALIFICATION_MODE,
                "runtime_source_commit": runtime_source_commit,
                "runtime_source_identity": runtime_source_identity,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "issue"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--live-result", type=Path, required=True)
        subparser.add_argument("--expected-live-sha256", required=True)
        subparser.add_argument("--candidate-so", type=Path, required=True)
        subparser.add_argument("--patch-source", type=Path, default=PATCH_SOURCE)
        subparser.add_argument("--expected-source-commit")
        subparser.add_argument("--runtime-source-commit")
        subparser.add_argument(
            "--candidate-selector", default="streamk_coop128"
        )
        subparser.add_argument(
            "--qualification-profile",
            choices=tuple(QUALIFICATION_PROFILES),
            default="full_vocab",
        )
        subparser.add_argument(
            "--draft-vocab-blocks",
            type=Path,
            default=DRAFT_VOCAB_BLOCKS_SOURCE,
        )
        subparser.add_argument(
            "--diagnostic-task-profile",
            choices=tuple(DIAGNOSTIC_TASK_PROFILES),
            default=DEFAULT_DIAGNOSTIC_TASK_PROFILE,
        )
        if command == "issue":
            subparser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--sidecar", type=Path, required=True)
    verify_parser.add_argument("--expected-sidecar-sha256", required=True)
    verify_parser.add_argument("--candidate-so", type=Path, required=True)
    verify_parser.add_argument("--patch-source", type=Path, default=PATCH_SOURCE)
    verify_parser.add_argument("--candidate-selector")
    verify_parser.add_argument(
        "--qualification-profile", choices=tuple(QUALIFICATION_PROFILES)
    )
    verify_parser.add_argument(
        "--draft-vocab-blocks",
        type=Path,
        default=DRAFT_VOCAB_BLOCKS_SOURCE,
    )
    verify_parser.add_argument(
        "--diagnostic-task-profile",
        choices=tuple(DIAGNOSTIC_TASK_PROFILES),
        default=DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    )
    source_parser = subparsers.add_parser("source-binding")
    source_parser.add_argument("--source-commit", required=True)
    source_parser.add_argument("--patch-source", type=Path, default=PATCH_SOURCE)
    source_parser.add_argument(
        "--candidate-selector", default="identity_onen_b1"
    )
    attestation_parser = subparsers.add_parser("attestation")
    attestation_parser.add_argument("--attestation", type=Path, required=True)
    attestation_parser.add_argument("--expected-sidecar-sha256", required=True)
    attestation_parser.add_argument(
        "--qualification-profile", choices=tuple(QUALIFICATION_PROFILES)
    )
    attestation_parser.add_argument(
        "--draft-vocab-blocks",
        type=Path,
        default=DRAFT_VOCAB_BLOCKS_SOURCE,
    )
    attestation_parser.add_argument(
        "--patch-source", type=Path, default=PATCH_SOURCE
    )
    attestation_parser.add_argument(
        "--diagnostic-task-profile",
        choices=tuple(DIAGNOSTIC_TASK_PROFILES),
        default=DEFAULT_DIAGNOSTIC_TASK_PROFILE,
    )
    args = parser.parse_args()

    if args.command == "validate":
        payload = validate_live_result(
            args.live_result,
            args.expected_live_sha256,
            args.candidate_so,
            args.patch_source,
            args.expected_source_commit,
            args.candidate_selector,
            args.qualification_profile,
            args.draft_vocab_blocks,
            args.diagnostic_task_profile,
            args.runtime_source_commit,
        )
    elif args.command == "issue":
        payload = issue_sidecar(
            args.live_result,
            args.expected_live_sha256,
            args.candidate_so,
            args.out,
            args.patch_source,
            args.expected_source_commit,
            args.candidate_selector,
            args.qualification_profile,
            args.draft_vocab_blocks,
            args.diagnostic_task_profile,
            args.runtime_source_commit,
        )
    elif args.command == "verify":
        payload = verify_sidecar(
            args.sidecar,
            args.expected_sidecar_sha256,
            args.candidate_so,
            args.patch_source,
            candidate_selector=args.candidate_selector,
            qualification_profile=args.qualification_profile,
            draft_vocab_blocks=args.draft_vocab_blocks,
            diagnostic_task_profile=args.diagnostic_task_profile,
        )
    elif args.command == "source-binding":
        payload = validate_source_commit_binding(
            args.source_commit,
            args.patch_source,
            args.candidate_selector,
        )
    else:
        payload = validate_production_attestation(
            args.attestation,
            args.expected_sidecar_sha256,
            args.qualification_profile,
            args.draft_vocab_blocks,
            args.patch_source,
            args.diagnostic_task_profile,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
