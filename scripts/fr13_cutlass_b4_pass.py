#!/usr/bin/env python3
"""Issue and verify the fixed32 B4 CUTLASS production credential."""

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


LIVE_SCHEMA = "fr13.fixed32.cutlass_persistent_b4_m128_live_gate.v1"
SIDECAR_SCHEMA = "fr13.fixed32.cutlass_b4.production_pass.v1"
K64_ROOT_LIVE_SCHEMA = "fr13.fixed32.cutlass_persistent_b4_m128_k64_root_live_gate.v1"
K64_ROOT_SIDECAR_SCHEMA = "fr13.fixed32.cutlass_b4.k64_root.production_pass.v1"
STATIC_LIVE_SCHEMA = "fr13.fixed32.cutlass_persistent_b4_m128_static_live_gate.v1"
STATIC_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_persistent_b4_m128_static_k64_root_live_gate.v1"
)
STATIC_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.m128_static.production_pass.v1"
)
STATIC_K64_ROOT_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.m128_static.k64_root.production_pass.v1"
)
IDENTITY_STOCKSHAPE_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stockshape_b4_live_gate.v1"
)
IDENTITY_STOCKSHAPE_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stockshape_b4_k64_root_live_gate.v1"
)
IDENTITY_STOCKSHAPE_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_stockshape.production_pass.v1"
)
IDENTITY_STOCKSHAPE_K64_ROOT_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_stockshape.k64_root.production_pass.v1"
)
IDENTITY_STOCKSHAPE_STAGE2_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stockshape_stage2_b4_live_gate.v1"
)
IDENTITY_STOCKSHAPE_STAGE2_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_stockshape_stage2_b4_k64_root_live_gate.v1"
)
IDENTITY_STOCKSHAPE_STAGE2_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_stockshape_stage2.production_pass.v1"
)
IDENTITY_STOCKSHAPE_STAGE2_K64_ROOT_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_stockshape_stage2.k64_root.production_pass.v1"
)
IDENTITY_DIVISOR_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_divisor_b4_live_gate.v1"
)
IDENTITY_DIVISOR_K64_ROOT_LIVE_SCHEMA = (
    "fr13.fixed32.cutlass_identity_divisor_b4_k64_root_live_gate.v1"
)
IDENTITY_DIVISOR_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_divisor.production_pass.v1"
)
IDENTITY_DIVISOR_K64_ROOT_SIDECAR_SCHEMA = (
    "fr13.fixed32.cutlass_b4.identity_divisor.k64_root.production_pass.v1"
)
ATTESTATION_SCHEMA = "fr13.fixed32.cutlass_streamk_binary.v2"
PATCH_SOURCE = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
PATCH_SOURCE_SHA256 = "656c53b20497fc08cc7fdfb18256235b07cfad9868fde2faa70e6b0b9dfca41a"
DRAFT_VOCAB_BLOCKS_SOURCE = Path("scripts/fr13_dvk_subset_blocks.json")
DRAFT_VOCAB_BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
PATCHED_DISPATCH_SHA256 = (
    "13debfa754beeb4a6ae9818612b4bf729619f0be03637372626de3778b2b3780"
)
STATIC_PATCH_SOURCE_SHA256 = (
    "977c0204d03d022bd3f4b745ad4a0bad8ec36d7bf82ac1c6f82aa42a62094fab"
)
STATIC_PATCHED_DISPATCH_SHA256 = (
    "446771039af31a2ae386b917540be2a018fdc8d947c001030696ec9a6608a4c4"
)
IDENTITY_STOCKSHAPE_PATCH_SOURCE_SHA256 = (
    "73d889292a812068593a26e83b320b8e12a27811ffba07e6f8532d0030f3cd90"
)
IDENTITY_STOCKSHAPE_PATCHED_DISPATCH_SHA256 = (
    "9bdd435d68cf914203faf4603c068e3df53e1b80efe3335d56671d1910ac45bf"
)
IDENTITY_STOCKSHAPE_STAGE2_PATCH_SOURCE_SHA256 = (
    "b4ec629ad903992030730ba15e24d9820cae771527953035c275402dd465fbec"
)
IDENTITY_STOCKSHAPE_STAGE2_PATCHED_DISPATCH_SHA256 = (
    "d21bff568fea29bc3a801d31b274f76692559aa22eab549fede9823fe5ee72c7"
)
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXPECTED_TASK_MARKERS = frozenset(
    f"swe_verified:{task_id}" for task_id in EXPECTED_TASK_IDS
)
EXPECTED_DRAFT_VOCAB_ROOT = 0
EXPECTED_DRAFT_VOCAB_K = 0
EXPECTED_MANDATORY_WEIGHT_BYTES = floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES
EXPECTED_MANDATORY_WEIGHT_FLOOR_MS = floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
EXPECTED_SLO_CAP_MS = floor.FULL_VOCAB_SLO_CAP_MS
K64_ROOT_MANDATORY_WEIGHT_BYTES = floor.FIXED32_MANDATORY_WEIGHT_BYTES
K64_ROOT_MANDATORY_WEIGHT_FLOOR_MS = floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS
K64_ROOT_SLO_CAP_MS = 137.6067177261
MAX_COMPARISONS = 320
QUALIFIED_FIXED32_MODES = ("tail6_fixed32", "hydra27_fixed32")
EXPECTED_PROJECTION_NK = (
    (5120, 6144),
    (5120, 17408),
    (14336, 5120),
    (16384, 5120),
    (34816, 5120),
)
CANDIDATE_CONTRACTS = {
    "identity_divisor_b4": {
        "diagnostic_selector": "identity_divisor_b4_byte_ab",
        "live_schemas": {
            "full_vocab": IDENTITY_DIVISOR_LIVE_SCHEMA,
            "k64_root": IDENTITY_DIVISOR_K64_ROOT_LIVE_SCHEMA,
        },
        "sidecar_schemas": {
            "full_vocab": IDENTITY_DIVISOR_SIDECAR_SCHEMA,
            "k64_root": IDENTITY_DIVISOR_K64_ROOT_SIDECAR_SCHEMA,
        },
        "binding_schemas": {
            "full_vocab": (
                "fr13.fixed32.cutlass_b4.identity_divisor.production_binding.v1"
            ),
            "k64_root": (
                "fr13.fixed32.cutlass_b4.identity_divisor.k64_root.production_binding.v1"
            ),
        },
        "production_authorized": False,
        "requires_resource_credential": False,
    },
    "identity_stockshape_b4": {
        "diagnostic_selector": "identity_stockshape_b4_byte_ab",
        "live_schemas": {
            "full_vocab": IDENTITY_STOCKSHAPE_LIVE_SCHEMA,
            "k64_root": IDENTITY_STOCKSHAPE_K64_ROOT_LIVE_SCHEMA,
        },
        "sidecar_schemas": {
            "full_vocab": IDENTITY_STOCKSHAPE_SIDECAR_SCHEMA,
            "k64_root": IDENTITY_STOCKSHAPE_K64_ROOT_SIDECAR_SCHEMA,
        },
        "binding_schemas": {
            "full_vocab": (
                "fr13.fixed32.cutlass_b4.identity_stockshape.production_binding.v1"
            ),
            "k64_root": (
                "fr13.fixed32.cutlass_b4.identity_stockshape.k64_root.production_binding.v1"
            ),
        },
        "production_authorized": False,
        "requires_resource_credential": False,
    },
    "identity_stockshape_stage2_b4": {
        "diagnostic_selector": "identity_stockshape_stage2_b4_byte_ab",
        "live_schemas": {
            "full_vocab": IDENTITY_STOCKSHAPE_STAGE2_LIVE_SCHEMA,
            "k64_root": IDENTITY_STOCKSHAPE_STAGE2_K64_ROOT_LIVE_SCHEMA,
        },
        "sidecar_schemas": {
            "full_vocab": IDENTITY_STOCKSHAPE_STAGE2_SIDECAR_SCHEMA,
            "k64_root": IDENTITY_STOCKSHAPE_STAGE2_K64_ROOT_SIDECAR_SCHEMA,
        },
        "binding_schemas": {
            "full_vocab": (
                "fr13.fixed32.cutlass_b4.identity_stockshape_stage2.production_binding.v1"
            ),
            "k64_root": (
                "fr13.fixed32.cutlass_b4.identity_stockshape_stage2.k64_root.production_binding.v1"
            ),
        },
        "production_authorized": False,
        "requires_resource_credential": False,
    },
    "persistent_b4_m128": {
        "diagnostic_selector": "persistent_b4_m128_byte_ab",
        "live_schemas": {
            "full_vocab": LIVE_SCHEMA,
            "k64_root": K64_ROOT_LIVE_SCHEMA,
        },
        "sidecar_schemas": {
            "full_vocab": SIDECAR_SCHEMA,
            "k64_root": K64_ROOT_SIDECAR_SCHEMA,
        },
        "binding_schemas": {
            "full_vocab": "fr13.fixed32.cutlass_b4.production_binding.v1",
            "k64_root": (
                "fr13.fixed32.cutlass_b4.k64_root.production_binding.v1"
            ),
        },
        "production_authorized": True,
        "requires_resource_credential": False,
    },
    "persistent_b4_m128_static": {
        "diagnostic_selector": "persistent_b4_m128_static_byte_ab",
        "live_schemas": {
            "full_vocab": STATIC_LIVE_SCHEMA,
            "k64_root": STATIC_K64_ROOT_LIVE_SCHEMA,
        },
        "sidecar_schemas": {
            "full_vocab": STATIC_SIDECAR_SCHEMA,
            "k64_root": STATIC_K64_ROOT_SIDECAR_SCHEMA,
        },
        "binding_schemas": {
            "full_vocab": (
                "fr13.fixed32.cutlass_b4.m128_static.production_binding.v1"
            ),
            "k64_root": (
                "fr13.fixed32.cutlass_b4.m128_static.k64_root.production_binding.v1"
            ),
        },
        "production_authorized": False,
        "requires_resource_credential": True,
    },
}
QUALIFICATION_PROFILES: dict[str, dict[str, object]] = {
    "full_vocab": {
        "live_schema": LIVE_SCHEMA,
        "sidecar_schema": SIDECAR_SCHEMA,
        "binding_schema": "fr13.fixed32.cutlass_b4.production_binding.v1",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": EXPECTED_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": EXPECTED_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": EXPECTED_SLO_CAP_MS,
        "requires_block_map": False,
    },
    "k64_root": {
        "live_schema": K64_ROOT_LIVE_SCHEMA,
        "sidecar_schema": K64_ROOT_SIDECAR_SCHEMA,
        "binding_schema": ("fr13.fixed32.cutlass_b4.k64_root.production_binding.v1"),
        "run_classification": ("real_swe_verified_exact4_b4_k64_root_byte_diagnostic"),
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "mandatory_weight_bytes": K64_ROOT_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": K64_ROOT_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": K64_ROOT_SLO_CAP_MS,
        "requires_block_map": True,
    },
}


def _candidate_contract(candidate_selector: str) -> dict[str, object]:
    try:
        return CANDIDATE_CONTRACTS[candidate_selector]
    except KeyError as error:
        raise QualificationError(
            f"CUTLASS B4 candidate selector mismatch: {candidate_selector!r}"
        ) from error


def _contract_profile_value(
    candidate_contract: dict[str, object],
    field: str,
    qualification_profile: str,
) -> str:
    values = candidate_contract.get(field)
    if not isinstance(values, dict):
        raise QualificationError(f"CUTLASS B4 candidate contract lacks {field}")
    value = values.get(qualification_profile)
    if not isinstance(value, str):
        raise QualificationError(
            f"CUTLASS B4 candidate contract lacks {field}.{qualification_profile}"
        )
    return value


def _candidate_source_hashes(candidate_selector: str) -> tuple[str, str]:
    if candidate_selector == "identity_divisor_b4":
        return (
            IDENTITY_STOCKSHAPE_PATCH_SOURCE_SHA256,
            IDENTITY_STOCKSHAPE_PATCHED_DISPATCH_SHA256,
        )
    if candidate_selector == "identity_stockshape_b4":
        return (
            IDENTITY_STOCKSHAPE_PATCH_SOURCE_SHA256,
            IDENTITY_STOCKSHAPE_PATCHED_DISPATCH_SHA256,
        )
    if candidate_selector == "identity_stockshape_stage2_b4":
        return (
            IDENTITY_STOCKSHAPE_STAGE2_PATCH_SOURCE_SHA256,
            IDENTITY_STOCKSHAPE_STAGE2_PATCHED_DISPATCH_SHA256,
        )
    if candidate_selector == "persistent_b4_m128":
        return PATCH_SOURCE_SHA256, PATCHED_DISPATCH_SHA256
    if candidate_selector == "persistent_b4_m128_static":
        return STATIC_PATCH_SOURCE_SHA256, STATIC_PATCHED_DISPATCH_SHA256
    raise QualificationError(
        f"CUTLASS B4 candidate selector mismatch: {candidate_selector!r}"
    )


class QualificationError(ValueError):
    """The CUTLASS B4 qualification chain is incomplete or inconsistent."""


def _qualification_profile(name: str) -> dict[str, object]:
    try:
        return QUALIFICATION_PROFILES[name]
    except KeyError as error:
        raise QualificationError(
            f"unsupported CUTLASS B4 qualification profile: {name!r}"
        ) from error


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
    path: Path,
    expected_sha256: str = PATCH_SOURCE_SHA256,
) -> dict[str, object]:
    info = _regular_file(path, "CUTLASS wave patch source")
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise QualificationError(
            f"CUTLASS wave patch source SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
    }


def _resource_binding(candidate: dict[str, object]) -> dict[str, object]:
    resource = candidate.get("resource_credential")
    if not isinstance(resource, dict):
        raise QualificationError(
            "static M128 candidate lacks its pinned resource credential"
        )
    return {
        "resource_credential_sha256": resource["sha256"],
        "resource_credential_schema": resource["schema"],
        "candidate_resource_records": resource["resource_records"],
        "candidate_registers_per_thread": resource["registers_per_thread"],
        "candidate_stack_bytes_per_thread": resource["stack_bytes_per_thread"],
        "candidate_local_bytes_per_thread": resource["local_bytes_per_thread"],
        "candidate_static_shared_bytes_per_cta": resource[
            "static_shared_bytes_per_cta"
        ],
        "candidate_constant0_bytes": resource["constant0_bytes"],
        "candidate_parameter_bytes": resource["parameter_bytes"],
        "candidate_threads_per_cta": resource["threads_per_cta"],
        "candidate_warps_per_cta": resource["warps_per_cta"],
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
    candidate_selector: str = "persistent_b4_m128",
    qualification_profile: str = "full_vocab",
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    fixed32_mode: str = "hydra27_fixed32",
    resource_credential: Path | None = None,
    expected_resource_credential_sha256: str | None = None,
) -> dict[str, Any]:
    if fixed32_mode not in QUALIFIED_FIXED32_MODES:
        raise QualificationError(
            f"unsupported CUTLASS B4 fixed32 mode: {fixed32_mode!r}"
        )
    candidate_contract = _candidate_contract(candidate_selector)
    profile = _qualification_profile(qualification_profile)
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected live-result SHA-256"
    )
    payload, raw = _read_json(live_result, "CUTLASS B4 live PASS")
    live_sha256 = hashlib.sha256(raw).hexdigest()
    if live_sha256 != expected_live_sha256:
        raise QualificationError(
            f"CUTLASS B4 live PASS SHA-256 mismatch: {live_sha256} != {expected_live_sha256}"
        )

    candidate = binary.verify_candidate(
        candidate_so,
        diagnostic_selector,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=(
            expected_resource_credential_sha256
        ),
    )
    patch_source_sha256, patched_dispatch_sha256 = _candidate_source_hashes(
        candidate_selector
    )
    patch = _validate_patch_source(patch_source, patch_source_sha256)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    expected_fields: dict[str, object] = {
        "schema": _contract_profile_value(
            candidate_contract, "live_schemas", qualification_profile
        ),
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 4,
        "task_ids": list(EXPECTED_TASK_IDS),
        "topology": fixed32_mode,
        "draft_vocab_root": profile["draft_vocab_root"],
        "draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "comparator_timing_eligible": False,
        "batch_size": 4,
        "concurrency": 4,
        "fixed_rows": 128,
        "eager_builder_capacity": 128,
        "candidate": candidate_selector,
        "diagnostic_selector": diagnostic_selector,
        "served_result": "stock",
        "production_enabled": False,
        "comparison_call_limit": MAX_COMPARISONS,
        "observed_m_values": [128],
        "observed_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": patch_source_sha256,
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": patched_dispatch_sha256,
        "errors": [],
    }
    if candidate_contract["requires_resource_credential"] is True:
        expected_fields.update(_resource_binding(candidate))
    if qualification_profile == "k64_root":
        assert block_map is not None
        expected_fields.update(
            {
                "qualification_profile": qualification_profile,
                "draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                "draft_vocab_blocks_sha256": block_map["sha256"],
            }
        )
    expected_fields["candidate_family"] = candidate["candidate_family"]
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"CUTLASS B4 live PASS {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    task_marker = payload.get("task_marker")
    if not isinstance(task_marker, str) or task_marker not in EXPECTED_TASK_MARKERS:
        raise QualificationError(
            "CUTLASS B4 live PASS task marker is not in the canonical exact4 set"
        )
    comparisons = payload.get("comparisons")
    if (
        isinstance(comparisons, bool)
        or not isinstance(comparisons, int)
        or comparisons < len(EXPECTED_PROJECTION_NK)
        or comparisons > MAX_COMPARISONS
    ):
        raise QualificationError("CUTLASS B4 live PASS comparison count is invalid")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("CUTLASS B4 live PASS source commit is invalid")
    if expected_source_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None:
            raise QualificationError("expected source commit is invalid")
        if source_commit != expected_source_commit:
            raise QualificationError(
                "CUTLASS B4 live PASS source commit is stale: "
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
        "schema": _contract_profile_value(
            candidate_contract, "sidecar_schemas", qualification_profile
        ),
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
        "patched_dispatch_sha256": patched_dispatch_sha256,
        "qualification_source_commit": source_commit,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_task_marker": task_marker,
        "real_task_arm_sha256": real_task_arm_sha256,
        "container_env_sha256": container_env_sha256,
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "qualified_topology": fixed32_mode,
        "qualified_comparison_call_limit": MAX_COMPARISONS,
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 128,
        "qualified_eager_builder_capacity": 128,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
    if candidate_contract["requires_resource_credential"] is True:
        result.update(_resource_binding(candidate))
    if qualification_profile == "k64_root":
        assert block_map is not None
        result.update(
            {
                "qualification_profile": qualification_profile,
                "qualified_draft_vocab_blocks": (DRAFT_VOCAB_BLOCKS_CONTAINER_PATH),
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
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
    candidate_selector: str = "persistent_b4_m128",
    qualification_profile: str = "full_vocab",
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
    fixed32_mode: str = "hydra27_fixed32",
    resource_credential: Path | None = None,
    expected_resource_credential_sha256: str | None = None,
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
        fixed32_mode,
        resource_credential,
        expected_resource_credential_sha256,
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
    fixed32_mode: str = "hydra27_fixed32",
    resource_credential: Path | None = None,
    expected_resource_credential_sha256: str | None = None,
) -> dict[str, Any]:
    if fixed32_mode not in QUALIFIED_FIXED32_MODES:
        raise QualificationError(
            f"unsupported CUTLASS B4 fixed32 mode: {fixed32_mode!r}"
        )
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(sidecar, "CUTLASS B4 production sidecar")
    sidecar_selector = payload.get("candidate_selector")
    if not isinstance(sidecar_selector, str):
        raise QualificationError("CUTLASS B4 production sidecar selector is invalid")
    candidate_contract = _candidate_contract(sidecar_selector)
    schema = payload.get("schema")
    if schema == _contract_profile_value(
        candidate_contract, "sidecar_schemas", "full_vocab"
    ):
        sidecar_profile = "full_vocab"
    elif schema == _contract_profile_value(
        candidate_contract, "sidecar_schemas", "k64_root"
    ):
        sidecar_profile = "k64_root"
    else:
        raise QualificationError("CUTLASS B4 production sidecar schema mismatch")
    if qualification_profile is not None and sidecar_profile != qualification_profile:
        raise QualificationError(
            "CUTLASS B4 production sidecar qualification-profile mismatch"
        )
    profile = _qualification_profile(sidecar_profile)
    if candidate_selector is not None and sidecar_selector != candidate_selector:
        raise QualificationError("CUTLASS B4 production sidecar selector mismatch")
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    if not isinstance(diagnostic_selector, str):
        raise QualificationError("CUTLASS B4 diagnostic selector contract is invalid")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sidecar_sha256:
        raise QualificationError(
            "CUTLASS B4 production sidecar SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sidecar_sha256}"
        )
    candidate = binary.verify_candidate(
        candidate_so,
        diagnostic_selector,
        resource_credential=resource_credential,
        expected_resource_credential_sha256=(
            expected_resource_credential_sha256
        ),
    )
    patch_source_sha256, patched_dispatch_sha256 = _candidate_source_hashes(
        sidecar_selector
    )
    patch = _validate_patch_source(patch_source, patch_source_sha256)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    required = {
        "schema": _contract_profile_value(
            candidate_contract, "sidecar_schemas", sidecar_profile
        ),
        "status": "QUALIFIED",
        "candidate_selector": sidecar_selector,
        "diagnostic_selector": diagnostic_selector,
        "candidate_family": candidate["candidate_family"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": patch["sha256"],
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": patched_dispatch_sha256,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "qualified_topology": fixed32_mode,
        "qualified_comparison_call_limit": MAX_COMPARISONS,
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 128,
        "qualified_eager_builder_capacity": 128,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
    if candidate_contract["requires_resource_credential"] is True:
        required.update(_resource_binding(candidate))
    if sidecar_profile == "k64_root":
        assert block_map is not None
        required.update(
            {
                "qualification_profile": sidecar_profile,
                "qualified_draft_vocab_blocks": (DRAFT_VOCAB_BLOCKS_CONTAINER_PATH),
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
            }
        )
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"CUTLASS B4 production sidecar {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    qualification_task_marker = payload.get("qualification_task_marker")
    if (
        not isinstance(qualification_task_marker, str)
        or qualification_task_marker not in EXPECTED_TASK_MARKERS
    ):
        raise QualificationError(
            "CUTLASS B4 production sidecar task marker is not canonical exact4"
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
    fixed32_mode: str = "hydra27_fixed32",
) -> dict[str, Any]:
    if fixed32_mode not in QUALIFIED_FIXED32_MODES:
        raise QualificationError(
            f"unsupported CUTLASS B4 fixed32 mode: {fixed32_mode!r}"
        )
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(attestation, "CUTLASS B4 binary attestation")
    candidate_selector = payload.get("selector")
    if not isinstance(candidate_selector, str):
        raise QualificationError("CUTLASS B4 binary attestation selector is invalid")
    candidate_contract = _candidate_contract(candidate_selector)
    if candidate_contract["production_authorized"] is not True:
        raise QualificationError(
            "static M128 production remains unavailable until Tail23 and Hydra27 "
            "raw-byte gates pass"
        )
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
                f"CUTLASS B4 binary attestation {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    for label, expected_path in (
        ("source", binary.CONTAINER_SOURCE),
        ("destination", binary.CONTAINER_DESTINATION),
    ):
        identity = payload.get(label)
        if not isinstance(identity, dict):
            raise QualificationError(f"CUTLASS B4 binary attestation lacks {label}")
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
                    f"CUTLASS B4 binary attestation {label}.{key} mismatch"
                )
    qualification = payload.get("qualification")
    if not isinstance(qualification, dict):
        raise QualificationError("CUTLASS B4 binary attestation lacks qualification")
    attested_profile = qualification.get("qualification_profile", "full_vocab")
    if not isinstance(attested_profile, str):
        raise QualificationError(
            "CUTLASS B4 attestation qualification profile is invalid"
        )
    if qualification_profile is not None and attested_profile != qualification_profile:
        raise QualificationError(
            "CUTLASS B4 attestation qualification-profile binding mismatch"
        )
    profile = _qualification_profile(attested_profile)
    block_map = (
        _validate_draft_vocab_blocks(draft_vocab_blocks)
        if profile["requires_block_map"]
        else None
    )
    if qualification.get("sidecar_sha256") != expected_sidecar_sha256:
        raise QualificationError("CUTLASS B4 attestation sidecar binding mismatch")
    if qualification.get("candidate_sha256") != candidate_sha256:
        raise QualificationError("CUTLASS B4 attestation candidate binding mismatch")
    patch_source_sha256, _ = _candidate_source_hashes(candidate_selector)
    if qualification.get("patch_source_sha256") != patch_source_sha256:
        raise QualificationError("CUTLASS B4 attestation patch-source binding mismatch")
    for key, expected in (
        ("qualified_draft_vocab_root", profile["draft_vocab_root"]),
        ("qualified_draft_vocab_k", profile["draft_vocab_k"]),
        ("qualified_topology", fixed32_mode),
        ("qualified_comparison_call_limit", MAX_COMPARISONS),
        ("qualified_eager_builder_capacity", 128),
        ("mandatory_weight_bytes", profile["mandatory_weight_bytes"]),
        (
            "mandatory_weight_floor_ms",
            profile["mandatory_weight_floor_ms"],
        ),
        ("one_sided_u95_cap_ms", profile["one_sided_u95_cap_ms"]),
    ):
        if qualification.get(key) != expected:
            raise QualificationError(f"CUTLASS B4 attestation {key} binding mismatch")
    if attested_profile == "k64_root":
        assert block_map is not None
        for key, expected in (
            ("qualification_profile", attested_profile),
            (
                "qualified_draft_vocab_blocks",
                DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
            ),
            (
                "qualified_draft_vocab_blocks_sha256",
                block_map["sha256"],
            ),
            ("qualified_fixed_rows", 128),
            (
                "qualified_projection_nk",
                [list(shape) for shape in EXPECTED_PROJECTION_NK],
            ),
        ):
            if qualification.get(key) != expected:
                raise QualificationError(
                    f"CUTLASS B4 attestation {key} binding mismatch"
                )
    qualification_task_marker = qualification.get("qualification_task_marker")
    if (
        not isinstance(qualification_task_marker, str)
        or qualification_task_marker not in EXPECTED_TASK_MARKERS
    ):
        raise QualificationError(
            "CUTLASS B4 attestation task marker is not canonical exact4"
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
            "CUTLASS B4 attestation qualification source commit is invalid"
        )
    _require_sha256(
        qualification.get("live_result_sha256"),
        "attestation live-result SHA-256",
    )
    result = {
        "schema": _contract_profile_value(
            candidate_contract, "binding_schemas", attested_profile
        ),
        "status": "BOUND",
        "selector": candidate_selector,
        "diagnostic_selector": candidate_contract["diagnostic_selector"],
        "candidate_family": candidate_family,
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": candidate_bytes,
        "patch_source_sha256": patch_source_sha256,
        "production_sidecar_sha256": expected_sidecar_sha256,
        "live_result_sha256": qualification["live_result_sha256"],
        "binary_attestation_sha256": hashlib.sha256(raw).hexdigest(),
        "qualification_source_commit": qualification_source_commit,
        "qualification_task_marker": qualification_task_marker,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "real_task_arm_sha256": real_task_arm_sha256,
        "container_env_sha256": container_env_sha256,
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "qualified_topology": fixed32_mode,
        "qualified_comparison_call_limit": MAX_COMPARISONS,
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 128,
        "qualified_eager_builder_capacity": 128,
        "installed_mode": "0555",
        "production_default_enabled": False,
    }
    if attested_profile == "k64_root":
        assert block_map is not None
        result.update(
            {
                "qualification_profile": attested_profile,
                "qualified_draft_vocab_blocks": (DRAFT_VOCAB_BLOCKS_CONTAINER_PATH),
                "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
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
        subparser.add_argument("--candidate-selector", default="persistent_b4_m128")
        subparser.add_argument("--resource-credential", type=Path)
        subparser.add_argument("--expected-resource-credential-sha256")
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
            "--fixed32-mode",
            choices=QUALIFIED_FIXED32_MODES,
            default="hydra27_fixed32",
        )
        if command == "issue":
            subparser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--sidecar", type=Path, required=True)
    verify_parser.add_argument("--expected-sidecar-sha256", required=True)
    verify_parser.add_argument("--candidate-so", type=Path, required=True)
    verify_parser.add_argument("--patch-source", type=Path, default=PATCH_SOURCE)
    verify_parser.add_argument("--candidate-selector")
    verify_parser.add_argument("--resource-credential", type=Path)
    verify_parser.add_argument("--expected-resource-credential-sha256")
    verify_parser.add_argument(
        "--qualification-profile", choices=tuple(QUALIFICATION_PROFILES)
    )
    verify_parser.add_argument(
        "--draft-vocab-blocks",
        type=Path,
        default=DRAFT_VOCAB_BLOCKS_SOURCE,
    )
    verify_parser.add_argument(
        "--fixed32-mode",
        choices=QUALIFIED_FIXED32_MODES,
        default="hydra27_fixed32",
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
        "--fixed32-mode",
        choices=QUALIFIED_FIXED32_MODES,
        default="hydra27_fixed32",
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
            args.fixed32_mode,
            args.resource_credential,
            args.expected_resource_credential_sha256,
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
            args.fixed32_mode,
            args.resource_credential,
            args.expected_resource_credential_sha256,
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
            fixed32_mode=args.fixed32_mode,
            resource_credential=args.resource_credential,
            expected_resource_credential_sha256=(
                args.expected_resource_credential_sha256
            ),
        )
    else:
        payload = validate_production_attestation(
            args.attestation,
            args.expected_sidecar_sha256,
            args.qualification_profile,
            args.draft_vocab_blocks,
            args.fixed32_mode,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
