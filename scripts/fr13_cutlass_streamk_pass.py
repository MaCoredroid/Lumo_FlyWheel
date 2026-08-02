#!/usr/bin/env python3
"""Issue and verify the fixed32 CUTLASS Stream-K production credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
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
K64_ROOT_SIDECAR_SCHEMA = "fr13.fixed32.cutlass_streamk.k64_root.production_pass.v1"
ATTESTATION_SCHEMA = "fr13.fixed32.cutlass_streamk_binary.v2"
PATCH_SOURCE = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
PATCH_SOURCE_SHA256 = "3132a7824feaabde09c249b29861fe8f4160d7b3c116b4423f39ecdd16554edc"
DRAFT_VOCAB_BLOCKS_SOURCE = Path("scripts/fr13_dvk_subset_blocks.json")
DRAFT_VOCAB_BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
PATCHED_DISPATCH_SHA256 = (
    "31e51c4a783b40d40454c0cb7e055c78ed19ad7247738f6ba606373880d2064a"
)
WIDE256_LIVE_SCHEMA = "fr13.fixed32.cutlass_streamk_wide256_live_gate.v1"
EXPECTED_TASK_IDS = ("astropy__astropy-12907",)
EXPECTED_TASK_MARKER = f"swe_verified:{EXPECTED_TASK_IDS[0]}"
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
}
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


def _candidate_contract(candidate_selector: str) -> dict[str, str]:
    try:
        return CANDIDATE_CONTRACTS[candidate_selector]
    except KeyError as error:
        raise QualificationError(
            f"Stream-K candidate selector mismatch: {candidate_selector!r}"
        ) from error


def _qualification_profile(
    candidate_selector: str, name: str
) -> dict[str, object]:
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
    }:
        raise QualificationError(
            "B1 k64_root qualification is restricted to wide256 or "
            "static-persistent stock-tile candidates"
        )
    result = dict(profile)
    if name == "full_vocab":
        result["live_schema"] = _candidate_contract(candidate_selector)["live_schema"]
    else:
        result["live_schema"] = _candidate_contract(candidate_selector)[
            "k64_root_live_schema"
        ]
    return result


class QualificationError(ValueError):
    """The Stream-K qualification chain is incomplete or inconsistent."""


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


def _validate_patch_source(path: Path) -> dict[str, object]:
    info = _regular_file(path, "Stream-K patch source")
    digest = sha256_file(path)
    if digest != PATCH_SOURCE_SHA256:
        raise QualificationError(
            f"Stream-K patch source SHA-256 mismatch: {digest} != {PATCH_SOURCE_SHA256}"
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


def validate_live_result(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = "streamk_coop128",
    qualification_profile: str = "full_vocab",
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
) -> dict[str, Any]:
    candidate_contract = _candidate_contract(candidate_selector)
    profile = _qualification_profile(candidate_selector, qualification_profile)
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

    candidate = binary.verify_candidate(candidate_so, diagnostic_selector)
    patch = _validate_patch_source(patch_source)
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
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_marker": EXPECTED_TASK_MARKER,
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
        "patch_source_sha256": PATCH_SOURCE_SHA256,
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        "errors": [],
    }
    if candidate_selector in {
        "streamk_force_wide256",
        "static_persistent_stocktile",
        "divisor_static_stocktile",
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
        "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        "qualification_source_commit": source_commit,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_task_marker": EXPECTED_TASK_MARKER,
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
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sidecar_sha256:
        raise QualificationError(
            "Stream-K production sidecar SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sidecar_sha256}"
        )
    candidate = binary.verify_candidate(candidate_so, diagnostic_selector)
    patch = _validate_patch_source(patch_source)
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
        "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_task_marker": EXPECTED_TASK_MARKER,
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
    return payload


def validate_production_attestation(
    attestation: Path,
    expected_sidecar_sha256: str,
    qualification_profile: str | None = None,
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(attestation, "Stream-K binary attestation")
    candidate_selector = payload.get("selector")
    if not isinstance(candidate_selector, str):
        raise QualificationError("Stream-K binary attestation selector is invalid")
    candidate_contract = _candidate_contract(candidate_selector)
    candidate_sha256, candidate_bytes, candidate_family = binary.candidate_identity(
        candidate_selector
    )
    required = {
        "schema": ATTESTATION_SCHEMA,
        "selector": candidate_selector,
        "installed_mode": "0555",
        "production_enabled": True,
    }
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
    if qualification.get("patch_source_sha256") != PATCH_SOURCE_SHA256:
        raise QualificationError("Stream-K attestation patch-source binding mismatch")
    for key, expected in (
        ("qualification_task_marker", EXPECTED_TASK_MARKER),
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
        "patch_source_sha256": PATCH_SOURCE_SHA256,
        "production_sidecar_sha256": expected_sidecar_sha256,
        "live_result_sha256": qualification["live_result_sha256"],
        "binary_attestation_sha256": hashlib.sha256(raw).hexdigest(),
        "qualification_source_commit": qualification_source_commit,
        "qualification_task_marker": EXPECTED_TASK_MARKER,
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
        )
    else:
        payload = validate_production_attestation(
            args.attestation,
            args.expected_sidecar_sha256,
            args.qualification_profile,
            args.draft_vocab_blocks,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
