#!/usr/bin/env python3
"""Reduce a real exact4 B1 stock/Stream-K full-wall timing pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import fr13_cutlass_wave_binary as binary
import fr13_cutlass_streamk_pass as qualification


SCHEMA = "fr13.fixed32.cutlass_streamk.b1_full_wall_timing_pair.v1"
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


class TimingError(ValueError):
    """The timing pair is incomplete or has mismatched provenance."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise TimingError(f"{label} is not a regular non-symlink file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimingError(f"{label} is not valid JSON") from error
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


def _validate_measure(record: dict[str, Any], label: str) -> dict[str, float]:
    required = {
        "schema": MEASURE_SCHEMA,
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": 4,
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
        "floor_ms",
        "floor_ratio",
    ):
        numeric[key] = _positive(record, key, label)
    return numeric


def reduce_pair(
    subset: Path,
    stock_measure: Path,
    candidate_measure: Path,
    production_binding: Path,
    candidate_so: Path,
) -> dict[str, Any]:
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
    expected_binding = {
        "schema": "fr13.fixed32.cutlass_streamk.production_binding.v1",
        "status": "BOUND",
        "selector": "streamk_coop128",
        "candidate_sha256": binary.CANDIDATE_SHA256,
        "candidate_bytes": binary.CANDIDATE_SIZE,
        "patch_source_sha256": qualification.PATCH_SOURCE_SHA256,
        "production_default_enabled": False,
    }
    for key, expected in expected_binding.items():
        if binding.get(key) != expected:
            raise TimingError(
                f"production binding {key} mismatch: "
                f"{binding.get(key)!r} != {expected!r}"
            )
    actual_candidate = binary.verify_candidate(candidate_so)
    if actual_candidate["sha256"] != binding["candidate_sha256"]:
        raise TimingError("production binding and candidate binary disagree")
    if not math.isclose(
        stock_values["floor_ms"],
        candidate_values["floor_ms"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise TimingError("stock and candidate optimistic floors differ")

    floor_ms = stock_values["floor_ms"]
    stock_wall = stock_values["step_wall_ms"]
    candidate_wall = candidate_values["step_wall_ms"]
    stock_tps = stock_values["measured_tps_fullstep_wall"]
    candidate_tps = candidate_values["measured_tps_fullstep_wall"]

    def arm(selector: str, values: dict[str, float]) -> dict[str, Any]:
        return {
            "selector": selector,
            "step_wall_ms": values["step_wall_ms"],
            "measured_tps_fullstep_wall": values["measured_tps_fullstep_wall"],
            "accepted_drafts_per_event": values["accept_per_event"],
            "committed_tokens_per_event": values["committed_per_event"],
            "s_fwd_gpu_ms_per_step": values["s_per_fwd_gpu"] * 1000.0,
            "drafter_gpu_ms_per_step": values["drafter_gpu_ms_per_step"],
            "committer_gpu_ms_per_step": values["committer_gpu_ms_per_step"],
            "step_wall_to_optimistic_floor_ratio": values["floor_ratio"],
            "wall_steps_measured": values["wall_steps_measured"],
            "events_per_step": values["events_per_step"],
        }

    result = {
        "schema": SCHEMA,
        "status": "complete",
        "run_classification": "real_swe_verified_exact4_b1_timing",
        "task_count": 4,
        "batch_size": 1,
        "concurrency": 1,
        "task_ids": sorted(EXPECTED_TASK_IDS),
        "decision_metric": "measured_tps_fullstep_wall",
        "stock_reference": arm("stock", stock_values),
        "candidate": {
            **arm("streamk_coop128", candidate_values),
            "candidate_sha256": binary.CANDIDATE_SHA256,
            "candidate_bytes": binary.CANDIDATE_SIZE,
            "patch_source_sha256": qualification.PATCH_SOURCE_SHA256,
            "production_binding_sha256": hashlib.sha256(binding_raw).hexdigest(),
            "live_result_sha256": binding["live_result_sha256"],
            "production_sidecar_sha256": binding["production_sidecar_sha256"],
        },
        "optimistic_mandatory_weight_floor_ms": floor_ms,
        "optimistic_floor_is_full_step_hardware_floor": False,
        "informational_1_15x_optimistic_floor_ms": floor_ms * 1.15,
        "candidate_to_stock_full_wall_tps_ratio": candidate_tps / stock_tps,
        "stock_to_candidate_step_wall_ratio": stock_wall / candidate_wall,
        "candidate_step_wall_delta_ms": candidate_wall - stock_wall,
        "formal_floor_acceptance_eligible": False,
        "formal_floor_acceptance_reason": (
            "paired exact4 Tail timing candidate only; the canonical Tail/Hydra "
            "floor gate and one-sided U95 procedure were not run"
        ),
        "production_default_enabled": False,
    }
    return result


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
    parser.add_argument("--production-binding", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = reduce_pair(
        args.subset,
        args.stock_measure,
        args.candidate_measure,
        args.production_binding,
        args.candidate_so,
    )
    _write(args.out, payload)
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
