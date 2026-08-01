#!/usr/bin/env python3
"""Reduce a real exact4 B1 stock/Stream-K full-wall timing pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any

import fr13_cutlass_streamk_pass as qualification
import fr13_cutlass_wave_binary as binary
import fr13_hardware_floor_ledger as floor


SCHEMA = "fr13.fixed32.cutlass_streamk.b1_full_wall_timing_pair.v2"
MEASURE_SCHEMA = "fr13.measure.deploy_speed.v1"
EXPECTED_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXPECTED_ENV = (
    "FR13_DRAFT_VOCAB_ROOT=0",
    "FR13_DRAFT_VOCAB_K=0",
)


class TimingError(ValueError):
    """The timing pair is incomplete or has mismatched provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise TimingError(f"{label} does not exist") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise TimingError(f"{label} is not a regular non-symlink file")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TimingError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise TimingError(f"non-finite JSON value: {value}")


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingError(f"{label} is not canonical ASCII JSON") from error
    if not isinstance(payload, dict):
        raise TimingError(f"{label} must contain a JSON object")
    return payload, raw


def _positive(record: dict[str, Any], key: str, label: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingError(f"{label} lacks numeric {key}")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise TimingError(f"{label} {key} is not finite and positive")
    return value


def _close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-9):
        raise TimingError(f"{label} mismatch: {actual!r} != {expected!r}")


def _validate_measure(record: dict[str, Any], label: str) -> dict[str, float]:
    required = {
        "schema": MEASURE_SCHEMA,
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": 4,
        "draft_vocab_k": 0,
        "draft_vocab_root": 0,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "floor_is_full_step_hardware_floor": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise TimingError(
                f"{label} {key} mismatch: {record.get(key)!r} != {expected!r}"
            )
    if sorted(record.get("task_instance_ids", [])) != sorted(EXPECTED_TASK_IDS):
        raise TimingError(f"{label} is not bound to the canonical exact4 task set")
    numeric = {}
    for key in (
        "measured_tps_fullstep_wall",
        "step_wall_ms",
        "accept_per_event",
        "committed_per_event",
        "wall_steps_measured",
        "events_per_step",
        "s_per_fwd_gpu",
        "drafter_gpu_ms_per_step",
        "committer_gpu_ms_per_step",
        "weight_floor_ms",
        "floor_ms",
        "floor_ratio",
    ):
        numeric[key] = _positive(record, key, label)
    _close(
        numeric["weight_floor_ms"],
        floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        f"{label} full-vocabulary weight floor",
    )
    _close(
        numeric["floor_ms"],
        floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        f"{label} active floor",
    )
    _close(
        numeric["floor_ratio"],
        numeric["step_wall_ms"] / numeric["floor_ms"],
        f"{label} floor ratio",
    )
    return numeric


def _validate_container_env(path: Path, label: str) -> str:
    _regular(path, label)
    raw = path.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeDecodeError as error:
        raise TimingError(f"{label} is not ASCII") from error
    for expected in EXPECTED_ENV:
        if lines.count(expected) != 1:
            raise TimingError(f"{label} lacks exact full-vocabulary pin {expected}")
    for prefix in ("FR13_DRAFT_VOCAB_ROOT=", "FR13_DRAFT_VOCAB_K="):
        matches = [line for line in lines if line.startswith(prefix)]
        if len(matches) != 1:
            raise TimingError(f"{label} has ambiguous {prefix[:-1]}")
    return hashlib.sha256(raw).hexdigest()


def reduce_pair(
    subset: Path,
    stock_measure: Path,
    candidate_measure: Path,
    stock_container_env: Path,
    candidate_container_env: Path,
    production_binding: Path,
    candidate_so: Path,
    source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise TimingError("timing source commit is invalid")
    _regular(subset, "exact4 subset")
    if _sha256(subset) != EXPECTED_SUBSET_SHA256:
        raise TimingError("canonical exact4 subset SHA-256 drift")
    subset_payload, _ = _load(subset, "exact4 subset")
    if sorted(subset_payload.get("instance_ids", [])) != sorted(EXPECTED_TASK_IDS):
        raise TimingError("canonical exact4 subset task IDs drift")
    stock, _ = _load(stock_measure, "stock full-wall measurement")
    candidate, _ = _load(candidate_measure, "candidate full-wall measurement")
    binding, binding_raw = _load(production_binding, "production binding")
    stock_values = _validate_measure(stock, "stock")
    candidate_values = _validate_measure(candidate, "candidate")
    stock_env_sha256 = _validate_container_env(
        stock_container_env, "stock container environment"
    )
    candidate_env_sha256 = _validate_container_env(
        candidate_container_env, "candidate container environment"
    )
    expected_binding = {
        "schema": "fr13.fixed32.cutlass_streamk.production_binding.v1",
        "status": "BOUND",
        "selector": "streamk_coop128",
        "candidate_sha256": binary.CANDIDATE_SHA256,
        "candidate_bytes": binary.CANDIDATE_SIZE,
        "patch_source_sha256": qualification.PATCH_SOURCE_SHA256,
        "qualification_source_commit": source_commit,
        "qualification_task_marker": qualification.EXPECTED_TASK_MARKER,
        "qualified_draft_vocab_root": 0,
        "qualified_draft_vocab_k": 0,
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
        "production_default_enabled": False,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise TimingError(
                f"production binding {key} mismatch: "
                f"{binding.get(key)!r} != {expected!r}"
            )
    for key in (
        "production_sidecar_sha256",
        "live_result_sha256",
        "binary_attestation_sha256",
        "real_task_arm_sha256",
        "container_env_sha256",
    ):
        value = binding.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise TimingError(f"production binding {key} is not SHA-256")
    actual_candidate = binary.verify_candidate(candidate_so)
    if actual_candidate["sha256"] != binding["candidate_sha256"]:
        raise TimingError("production binding and candidate binary disagree")

    stock_wall = stock_values["step_wall_ms"]
    candidate_wall = candidate_values["step_wall_ms"]
    stock_tps = stock_values["measured_tps_fullstep_wall"]
    candidate_tps = candidate_values["measured_tps_fullstep_wall"]

    def arm(
        selector: str, values: dict[str, float], container_env_sha256: str
    ) -> dict[str, Any]:
        return {
            "selector": selector,
            "step_wall_ms": values["step_wall_ms"],
            "measured_tps_fullstep_wall": values["measured_tps_fullstep_wall"],
            "accepted_drafts_per_event": values["accept_per_event"],
            "committed_tokens_per_event": values["committed_per_event"],
            "s_fwd_gpu_ms_per_step": values["s_per_fwd_gpu"] * 1000.0,
            "drafter_gpu_ms_per_step": values["drafter_gpu_ms_per_step"],
            "committer_gpu_ms_per_step": values["committer_gpu_ms_per_step"],
            "step_wall_to_mandatory_weight_floor_ratio": values["floor_ratio"],
            "wall_steps_measured": values["wall_steps_measured"],
            "events_per_step": values["events_per_step"],
            "container_env_sha256": container_env_sha256,
        }

    return {
        "schema": SCHEMA,
        "status": "complete",
        "run_classification": "real_swe_verified_exact4_b1_timing",
        "task_count": 4,
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": sorted(EXPECTED_TASK_IDS),
        "source_commit": source_commit,
        "decision_metric": "measured_tps_fullstep_wall",
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "stock_reference": arm("stock", stock_values, stock_env_sha256),
        "candidate": {
            **arm("streamk_coop128", candidate_values, candidate_env_sha256),
            "candidate_sha256": binary.CANDIDATE_SHA256,
            "candidate_bytes": binary.CANDIDATE_SIZE,
            "patch_source_sha256": qualification.PATCH_SOURCE_SHA256,
            "production_binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
            "live_result_sha256": binding["live_result_sha256"],
            "production_sidecar_sha256": binding[
                "production_sidecar_sha256"
            ],
            "real_task_arm_sha256": binding["real_task_arm_sha256"],
        },
        "mandatory_weight_bytes": floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": floor.FULL_VOCAB_SLO_CAP_MS,
        "mandatory_weight_floor_is_complete_step_floor": False,
        "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
        "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
        "candidate_step_wall_delta_ms": candidate_wall - stock_wall,
        "comparator_gate_timing_eligible": False,
        "timing_claim_source": "paired exact4 real SWE-Verified full-wall arms",
        "formal_floor_acceptance_eligible": False,
        "formal_floor_acceptance_reason": (
            "paired exact4 Tail timing candidate only; the canonical Tail/Hydra "
            "one-sided U95 floor gate was not run"
        ),
        "production_default_enabled": False,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--stock-measure", type=Path, required=True)
    parser.add_argument("--candidate-measure", type=Path, required=True)
    parser.add_argument("--stock-container-env", type=Path, required=True)
    parser.add_argument("--candidate-container-env", type=Path, required=True)
    parser.add_argument("--production-binding", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = reduce_pair(
        args.subset,
        args.stock_measure,
        args.candidate_measure,
        args.stock_container_env,
        args.candidate_container_env,
        args.production_binding,
        args.candidate_so,
        args.source_commit,
    )
    _write(args.out, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
