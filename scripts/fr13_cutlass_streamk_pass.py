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
ATTESTATION_SCHEMA = "fr13.fixed32.cutlass_streamk_binary.v2"
PATCH_SOURCE = Path("scripts/fr13_patch_cutlass_fixed32_wave.py")
PATCH_SOURCE_SHA256 = "656c53b20497fc08cc7fdfb18256235b07cfad9868fde2faa70e6b0b9dfca41a"
VLLM_BASE_COMMIT = "fe9c3d6c5f66c873d196800384ed6880687b9e52"
PATCHED_DISPATCH_SHA256 = (
    "13debfa754beeb4a6ae9818612b4bf729619f0be03637372626de3778b2b3780"
)
WIDE256_LIVE_SCHEMA = "fr13.fixed32.cutlass_streamk_wide256_live_gate.v1"
EXPECTED_TASK_IDS = ("astropy__astropy-12907",)
EXPECTED_TASK_MARKER = f"swe_verified:{EXPECTED_TASK_IDS[0]}"
EXPECTED_DRAFT_VOCAB_ROOT = 0
EXPECTED_DRAFT_VOCAB_K = 0
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
        "diagnostic_selector": "streamk_force_wide256_byte_ab",
    },
}


def _candidate_contract(candidate_selector: str) -> dict[str, str]:
    try:
        return CANDIDATE_CONTRACTS[candidate_selector]
    except KeyError as error:
        raise QualificationError(
            f"Stream-K candidate selector mismatch: {candidate_selector!r}"
        ) from error


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


def validate_live_result(
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    patch_source: Path = PATCH_SOURCE,
    expected_source_commit: str | None = None,
    candidate_selector: str = "streamk_coop128",
) -> dict[str, Any]:
    candidate_contract = _candidate_contract(candidate_selector)
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
    expected_fields: dict[str, object] = {
        "schema": candidate_contract["live_schema"],
        "status": "pass",
        "run_classification": "one_real_swe_verified_b1_byte_diagnostic",
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_marker": EXPECTED_TASK_MARKER,
        "draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
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
    if candidate_selector == "streamk_force_wide256":
        expected_fields["candidate_family"] = candidate["candidate_family"]
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
        or comparisons > 256
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
    return {
        "schema": SIDECAR_SCHEMA,
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
        "qualified_draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "qualified_draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
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
    candidate_selector: str = "streamk_coop128",
) -> dict[str, Any]:
    payload = validate_live_result(
        live_result,
        expected_live_sha256,
        candidate_so,
        patch_source,
        expected_source_commit,
        candidate_selector,
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
    diagnostic_selector = candidate_contract["diagnostic_selector"]
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sidecar_sha256:
        raise QualificationError(
            "Stream-K production sidecar SHA-256 mismatch: "
            f"{actual_sha256} != {expected_sidecar_sha256}"
        )
    candidate = binary.verify_candidate(candidate_so, diagnostic_selector)
    patch = _validate_patch_source(patch_source)
    required = {
        "schema": SIDECAR_SCHEMA,
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
        "qualified_draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "qualified_draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
        "qualified_projection_nk": [list(shape) for shape in EXPECTED_PROJECTION_NK],
        "qualified_fixed_rows": 32,
        "served_result_during_qualification": "stock",
        "production_default_enabled": False,
    }
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
    if qualification.get("sidecar_sha256") != expected_sidecar_sha256:
        raise QualificationError("Stream-K attestation sidecar binding mismatch")
    if qualification.get("candidate_sha256") != candidate_sha256:
        raise QualificationError("Stream-K attestation candidate binding mismatch")
    if qualification.get("patch_source_sha256") != PATCH_SOURCE_SHA256:
        raise QualificationError("Stream-K attestation patch-source binding mismatch")
    for key, expected in (
        ("qualification_task_marker", EXPECTED_TASK_MARKER),
        ("qualified_draft_vocab_root", EXPECTED_DRAFT_VOCAB_ROOT),
        ("qualified_draft_vocab_k", EXPECTED_DRAFT_VOCAB_K),
        ("mandatory_weight_bytes", floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES),
        (
            "mandatory_weight_floor_ms",
            floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        ),
        ("one_sided_u95_cap_ms", floor.FULL_VOCAB_SLO_CAP_MS),
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
        "schema": "fr13.fixed32.cutlass_streamk.production_binding.v1",
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
        "qualified_draft_vocab_root": EXPECTED_DRAFT_VOCAB_ROOT,
        "qualified_draft_vocab_k": EXPECTED_DRAFT_VOCAB_K,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
        "installed_mode": "0555",
        "production_default_enabled": False,
    }
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
        if command == "issue":
            subparser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--sidecar", type=Path, required=True)
    verify_parser.add_argument("--expected-sidecar-sha256", required=True)
    verify_parser.add_argument("--candidate-so", type=Path, required=True)
    verify_parser.add_argument("--patch-source", type=Path, default=PATCH_SOURCE)
    verify_parser.add_argument("--candidate-selector")
    attestation_parser = subparsers.add_parser("attestation")
    attestation_parser.add_argument("--attestation", type=Path, required=True)
    attestation_parser.add_argument("--expected-sidecar-sha256", required=True)
    args = parser.parse_args()

    if args.command == "validate":
        payload = validate_live_result(
            args.live_result,
            args.expected_live_sha256,
            args.candidate_so,
            args.patch_source,
            args.expected_source_commit,
            args.candidate_selector,
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
        )
    elif args.command == "verify":
        payload = verify_sidecar(
            args.sidecar,
            args.expected_sidecar_sha256,
            args.candidate_so,
            args.patch_source,
            candidate_selector=args.candidate_selector,
        )
    else:
        payload = validate_production_attestation(
            args.attestation, args.expected_sidecar_sha256
        )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
