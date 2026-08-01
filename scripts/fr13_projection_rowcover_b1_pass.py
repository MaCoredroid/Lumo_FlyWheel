#!/usr/bin/env python3
"""Issue and verify the exact-M32 K64 projection row-cover credential."""

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


LIVE_SCHEMA = "fr13.fixed32.cutlass_static_persistent_k64_root_live_gate.v1"
SIDECAR_SCHEMA = "fr13.fixed32.projection_rowcover_b1.k64_root.production_pass.v1"
BINDING_SCHEMA = "fr13.fixed32.projection_rowcover_b1.k64_root.production_binding.v1"
ATTESTATION_SCHEMA = "fr13.fixed32.cutlass_streamk_binary.v2"
PATCH_SOURCE = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
PATCH_SOURCE_SHA256 = "32ee5747eeff597f7eacec530f86658ba26b6fe8560591c21c305e594953935a"
PATCHED_DISPATCH_SHA256 = (
    "ba18f08dcbd17a52c1b7293be0cc6eb4ee57176388d4e2ccba9bfb62c9b31c45"
)
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
DRAFT_VOCAB_BLOCKS_SOURCE = Path("scripts/fr13_dvk_subset_blocks.json")
DRAFT_VOCAB_BLOCKS_CONTAINER_PATH = "/workspace/scripts/fr13_dvk_subset_blocks.json"
DRAFT_VOCAB_BLOCKS_SHA256 = (
    "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff"
)
CANDIDATE_SELECTOR = "static_persistent_stocktile"
DIAGNOSTIC_SELECTOR = "static_persistent_stocktile_byte_ab"
QUALIFICATION_PROFILE = "k64_root"
EXPECTED_TASK_IDS = ("astropy__astropy-12907",)
EXPECTED_TASK_MARKER = f"swe_verified:{EXPECTED_TASK_IDS[0]}"
EXPECTED_PROJECTION_NK = (
    (5120, 6144),
    (5120, 17408),
    (14336, 5120),
    (16384, 5120),
    (34816, 5120),
)
MAX_COMPARISONS = 320
EXPECTED_DRAFT_VOCAB_ROOT = 1
EXPECTED_DRAFT_VOCAB_K = 65_536
EXPECTED_SLO_CAP_MS = 137.6067177261


class QualificationError(ValueError):
    """The projection row-cover qualification chain is inconsistent."""


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


def _validate_exact_file(
    path: Path, label: str, expected_sha256: str
) -> dict[str, object]:
    info = _regular_file(path, label)
    digest = sha256_file(path)
    if digest != expected_sha256:
        raise QualificationError(
            f"{label} SHA-256 mismatch: {digest} != {expected_sha256}"
        )
    return {
        "path": str(path.resolve(strict=True)),
        "bytes": info.st_size,
        "sha256": digest,
        "regular": True,
        "symlink": False,
    }


def _require_selector(candidate_selector: str) -> None:
    if candidate_selector != CANDIDATE_SELECTOR:
        raise QualificationError(
            f"projection row-cover selector mismatch: {candidate_selector!r}"
        )


def validate_live_result(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = CANDIDATE_SELECTOR,
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
) -> dict[str, Any]:
    _require_selector(candidate_selector)
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "expected live-result SHA-256"
    )
    payload, raw = _read_json(live_result, "projection row-cover B1 live PASS")
    live_sha256 = hashlib.sha256(raw).hexdigest()
    if live_sha256 != expected_live_sha256:
        raise QualificationError(
            "projection row-cover live PASS SHA-256 mismatch: "
            f"{live_sha256} != {expected_live_sha256}"
        )

    candidate = binary.verify_candidate(candidate_so, DIAGNOSTIC_SELECTOR)
    patch = _validate_exact_file(
        patch_source, "projection row-cover patch source", PATCH_SOURCE_SHA256
    )
    block_map = _validate_exact_file(
        draft_vocab_blocks,
        "draft-vocabulary block map",
        DRAFT_VOCAB_BLOCKS_SHA256,
    )
    expected_fields: dict[str, object] = {
        "schema": LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_b1_k64_root_byte_diagnostic",
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_marker": EXPECTED_TASK_MARKER,
        "qualification_profile": QUALIFICATION_PROFILE,
        "draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": DRAFT_VOCAB_BLOCKS_SHA256,
        "mandatory_weight_bytes": floor.FIXED32_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": EXPECTED_SLO_CAP_MS,
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": CANDIDATE_SELECTOR,
        "diagnostic_selector": DIAGNOSTIC_SELECTOR,
        "served_result": "stock",
        "production_enabled": False,
        "observed_m_values": [32],
        "observed_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": candidate["candidate_family"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": PATCH_SOURCE_SHA256,
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        "errors": [],
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"projection row-cover live PASS {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    comparisons = payload.get("comparisons")
    if (
        isinstance(comparisons, bool)
        or not isinstance(comparisons, int)
        or comparisons < len(EXPECTED_PROJECTION_NK)
        or comparisons > MAX_COMPARISONS
    ):
        raise QualificationError("projection row-cover comparison count is invalid")
    source_commit = payload.get("source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("projection row-cover source commit is invalid")
    if expected_source_commit is not None:
        if re.fullmatch(r"[0-9a-f]{40}", expected_source_commit) is None:
            raise QualificationError("expected source commit is invalid")
        if source_commit != expected_source_commit:
            raise QualificationError(
                "projection row-cover source commit is stale: "
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
    return {
        "schema": SIDECAR_SCHEMA,
        "status": "QUALIFIED",
        "candidate_selector": CANDIDATE_SELECTOR,
        "diagnostic_selector": DIAGNOSTIC_SELECTOR,
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
        "qualification_profile": QUALIFICATION_PROFILE,
        "qualified_draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "qualified_draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "qualified_draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
        "mandatory_weight_bytes": floor.FIXED32_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": EXPECTED_SLO_CAP_MS,
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 32,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }


def issue_sidecar(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    output: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = CANDIDATE_SELECTOR,
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
) -> dict[str, Any]:
    payload = validate_live_result(
        live_result,
        expected_live_sha256,
        candidate_so,
        patch_source,
        expected_source_commit,
        candidate_selector,
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
    draft_vocab_blocks: Path = DRAFT_VOCAB_BLOCKS_SOURCE,
) -> dict[str, Any]:
    if candidate_selector is not None:
        _require_selector(candidate_selector)
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "expected production-sidecar SHA-256"
    )
    payload, raw = _read_json(sidecar, "projection row-cover production sidecar")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sidecar_sha256:
        raise QualificationError(
            "projection row-cover sidecar SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sidecar_sha256}"
        )
    candidate = binary.verify_candidate(candidate_so, DIAGNOSTIC_SELECTOR)
    patch = _validate_exact_file(
        patch_source, "projection row-cover patch source", PATCH_SOURCE_SHA256
    )
    block_map = _validate_exact_file(
        draft_vocab_blocks,
        "draft-vocabulary block map",
        DRAFT_VOCAB_BLOCKS_SHA256,
    )
    required = {
        "schema": SIDECAR_SCHEMA,
        "status": "QUALIFIED",
        "candidate_selector": CANDIDATE_SELECTOR,
        "diagnostic_selector": DIAGNOSTIC_SELECTOR,
        "candidate_family": candidate["candidate_family"],
        "candidate_sha256": candidate["sha256"],
        "candidate_bytes": candidate["bytes"],
        "patch_source_sha256": patch["sha256"],
        "vllm_base_commit": VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": PATCHED_DISPATCH_SHA256,
        "qualification_task_ids": list(EXPECTED_TASK_IDS),
        "qualification_task_marker": EXPECTED_TASK_MARKER,
        "qualification_profile": QUALIFICATION_PROFILE,
        "qualified_draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "qualified_draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "qualified_draft_vocab_blocks": DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "qualified_draft_vocab_blocks_sha256": block_map["sha256"],
        "mandatory_weight_bytes": floor.FIXED32_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FIXED32_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": EXPECTED_SLO_CAP_MS,
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 32,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise QualificationError(
                f"projection row-cover sidecar {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    for key in (
        "live_result_sha256",
        "binary_attestation_sha256",
        "real_task_arm_sha256",
        "container_env_sha256",
    ):
        _require_sha256(payload.get(key), f"sidecar {key}")
    source_commit = payload.get("qualification_source_commit")
    if (
        not isinstance(source_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
    ):
        raise QualificationError("sidecar qualification source commit is invalid")
    return payload


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
        subparser.add_argument("--candidate-selector", default=CANDIDATE_SELECTOR)
        subparser.add_argument(
            "--draft-vocab-blocks", type=Path, default=DRAFT_VOCAB_BLOCKS_SOURCE
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
        "--draft-vocab-blocks", type=Path, default=DRAFT_VOCAB_BLOCKS_SOURCE
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
            args.draft_vocab_blocks,
        )
    else:
        payload = verify_sidecar(
            args.sidecar,
            args.expected_sidecar_sha256,
            args.candidate_so,
            args.patch_source,
            candidate_selector=args.candidate_selector,
            draft_vocab_blocks=args.draft_vocab_blocks,
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
