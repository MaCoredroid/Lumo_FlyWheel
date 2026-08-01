#!/usr/bin/env python3
"""Reduce the paired one-task B1 CFWD all-parent timing diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fr13_hardware_floor_ledger import (  # noqa: E402
    BANDWIDTH_BYTES_PER_S,
    FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
    FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
)


SCHEMA = "fr13.fixed32.cfwd_all_parent_b1.full_wall_timing_pair.v1"
CANDIDATE_BASE_COMMIT = "f19e90053cfe414cafc76a2ffa3326a589da5e1e"
CANDIDATE = "fixed32_all_parent_commit_v2"
TASK_ID = "astropy__astropy-12907"
SUBSET_SHA256 = "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
LIVE_PASS_SHA256 = "b7c8f4e7f8cf3e2619d458b3ec3e5e1ffdcb5a15a2938aa18c6dda936b3c45e3"
SOURCE_CONTRACT_SCHEMA = "fr13-fixed32-taw-all-parent-v4"
SOURCE_CONTRACT_SHA256 = (
    "51541928c3a758fdac34a70fe46b97753ffc1b6e9f3e5fe470c4b34a96515dc4"
)
REFERENCE_ROUTE = "fixed32_pytorch_exact_float_triton_integer_commit"
PRODUCTION_ROUTE = "fixed32_native_precompute_production_candidate_return"
CENSUS_SCHEMA = "fr13-fixed32-work-census-v9"
CENSUS_TERMINAL_SCHEMA = "fr13-fixed32-work-census-terminal-v9"
FULL_VOCAB_CAP_MS = float(
    (
        Decimal(FULL_VOCAB_MANDATORY_WEIGHT_BYTES)
        * Decimal(1_000)
        / Decimal(BANDWIDTH_BYTES_PER_S)
        * Decimal("1.15")
    ).quantize(Decimal("0.000000001"))
)

_PASS_EXPECTED = {
    "schema": "fr13.fixed32.taw_native_precompute.live_pass.v1",
    "status": "pass",
    "candidate": CANDIDATE,
    "source_contract_schema": SOURCE_CONTRACT_SCHEMA,
    "source_contract_sha256": SOURCE_CONTRACT_SHA256,
    "task_marker": f"swe_verified:{TASK_ID}",
    "mode": "hydra27_fixed32",
    "batch_size": 1,
    "covered_batches": [1],
    "geometry": {
        "accepted_path_capacity": 16,
        "fanout": 3,
        "output_capacity": 32,
        "physical_drafts": 31,
        "physical_rows": 32,
        "walk_cap": 12,
    },
    "probability_mismatches": 0,
    "product_mismatches": 0,
    "evidence_route": "full_graph_replay",
    "reference_returned": True,
    "candidate_returned": False,
}

_REQUIRED_ENV = {
    "FR10_METRICS": "0",
    "FR13_CFWD_GPU_TIMER": "1",
    "FR13_DFWD_GPU_TIMER": "1",
    "FR13_DEVICE_MULTIDRAFT": "1",
    "FR13_DRAFT_VOCAB_K": "0",
    "FR13_DRAFT_VOCAB_ROOT": "0",
    "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB": "0",
    "FR13_DRAFT_HEAD_PAD_ROWS": "0",
    "FR13_DFWD_UNIFIED_BM8_LIVE_AB": "0",
    "FR13_DFWD_UNIFIED_BM8_PRODUCTION": "0",
    "FR13_FA2_QROW16_LIVE_PAGED_AB": "0",
    "FR13_FA2_QROW16_PRODUCTION": "0",
    "FR13_FIXED32_BATCH_GDN_BV_CANDIDATE": "",
    "FR13_FIXED32_BATCH_GDN_BV_PRODUCTION": "",
    "FR13_FIXED32_CUTLASS_WAVE": "stock",
    "FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "",
    "FR13_FIXED32_GDN_PATH_BV_PRODUCTION": "",
    "FR13_FIXED32_MODE": "hydra27_fixed32",
    "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE": "0",
    "FR13_FIXED32_WORK_CENSUS": "1",
    "FR13_SFWD_GPU_TIMER": "1",
}

_VOLATILE_ENV = {
    "FR13_CFWD_GPU_TIMER_JSON",
    "FR13_DFWD_GPU_TIMER_JSON",
    "FR13_RUN_DIR",
    "FR13_SFWD_GPU_TIMER_JSON",
    "HOSTNAME",
    "LUMO_NSYS_OUTPUT",
}


class TimingError(ValueError):
    """The paired timing evidence is incomplete, stale, or mismatched."""


def _duplicate_checked_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise TimingError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise TimingError(f"non-finite JSON constant: {value}")


def _regular(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
    except OSError as error:
        raise TimingError(f"{label} is unavailable: {error}") from error
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise TimingError(f"{label} must be a regular non-symlink file")
    try:
        return path.read_bytes()
    except OSError as error:
        raise TimingError(f"{label} is unreadable: {error}") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_duplicate_checked_object,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TimingError) as error:
        raise TimingError(f"{label} is not strict ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise TimingError(f"{label} must contain a JSON object")
    return payload, raw


def validate_live_pass(path: Path) -> dict[str, Any]:
    payload, raw = _load_json(path, "curated CFWD live PASS")
    if _sha256(raw) != LIVE_PASS_SHA256:
        raise TimingError("curated CFWD live PASS SHA-256 drifted")
    if set(payload) != set(_PASS_EXPECTED):
        raise TimingError("curated CFWD live PASS key set drifted")
    for key, expected in _PASS_EXPECTED.items():
        if payload.get(key) != expected:
            raise TimingError(
                f"curated CFWD live PASS {key} mismatch: "
                f"{payload.get(key)!r} != {expected!r}"
            )
    return {
        "sha256": LIVE_PASS_SHA256,
        "candidate": CANDIDATE,
        "source_contract_schema": SOURCE_CONTRACT_SCHEMA,
        "source_contract_sha256": SOURCE_CONTRACT_SHA256,
        "task_marker": payload["task_marker"],
        "qualified_batch_size": payload["batch_size"],
        "evidence_route": payload["evidence_route"],
    }


def _finite(record: dict[str, Any], key: str, *, positive: bool = True) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimingError(f"{key} is missing from full-wall timing evidence")
    result = float(value)
    if (
        not math.isfinite(result)
        or (positive and result <= 0)
        or (not positive and result < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise TimingError(f"{key} is not finite and {qualifier}")
    return result


def _validate_measure(record: dict[str, Any], label: str) -> dict[str, float]:
    if (
        record.get("schema") != "fr13.measure.deploy_speed.v1"
        or record.get("regime") != "deployment"
        or record.get("instrument") != "OFF"
        or record.get("batch_size") != 1
        or record.get("n_tasks") != 1
        or record.get("task_instance_ids") != [TASK_ID]
        or record.get("mandatory_weight_bytes") != FULL_VOCAB_MANDATORY_WEIGHT_BYTES
        or record.get("floor_is_full_step_hardware_floor") is not False
    ):
        raise TimingError(f"{label} deploy-speed provenance is not one-task B1 K=0")
    engagement = record.get("engagement")
    if (
        not isinstance(engagement, dict)
        or engagement.get("engaged") is not True
        or float(engagement.get("tok_per_draft", -1)) != 31.0
        or float(engagement.get("expected_tok_per_draft", -1)) != 31.0
    ):
        raise TimingError(f"{label} fixed32 speculative route was not engaged")

    values = {
        "measured_tps_fullstep_wall": _finite(record, "measured_tps_fullstep_wall"),
        "step_wall_ms": _finite(record, "step_wall_ms"),
        "accept_per_event": _finite(record, "accept_per_event", positive=False),
        "committed_per_event": _finite(record, "committed_per_event"),
        "wall_steps_measured": _finite(record, "wall_steps_measured"),
        "events_per_step": _finite(record, "events_per_step"),
        "s_fwd_gpu_ms_per_event": _finite(record, "s_per_fwd_gpu") * 1_000.0,
        "drafter_gpu_ms_per_event": _finite(record, "drafter_gpu_ms_per_step"),
        "cfwd_gpu_ms_per_event": _finite(record, "committer_gpu_ms_per_step"),
        "floor_ms": _finite(record, "floor_ms"),
        "weight_floor_ms": _finite(record, "weight_floor_ms"),
        "floor_ratio": _finite(record, "floor_ratio"),
    }
    if not math.isclose(values["events_per_step"], 1.0, abs_tol=1e-9):
        raise TimingError(f"{label} B1 events_per_step is not exactly one")
    if not math.isclose(
        values["committed_per_event"],
        values["accept_per_event"] + 1.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise TimingError(f"{label} committed/event does not equal accepted+bonus")
    expected_tps = values["committed_per_event"] * 1_000.0 / values["step_wall_ms"]
    if not math.isclose(
        values["measured_tps_fullstep_wall"], expected_tps, rel_tol=1e-9
    ):
        raise TimingError(f"{label} full-wall TPS is inconsistent with step wall")
    for key in ("floor_ms", "weight_floor_ms"):
        if not math.isclose(
            values[key],
            FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise TimingError(f"{label} {key} is not the full-vocabulary floor")
    if not math.isclose(
        values["floor_ratio"],
        values["step_wall_ms"] / values["floor_ms"],
        rel_tol=1e-9,
    ):
        raise TimingError(f"{label} floor ratio is inconsistent")
    if values["accept_per_event"] > 31.0:
        raise TimingError(f"{label} accepted drafts/event exceeds physical drafts")
    return values


def _load_env(path: Path, label: str) -> tuple[dict[str, str], str]:
    raw = _regular(path, label)
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise TimingError(f"{label} is not UTF-8") from error
    result: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            raise TimingError(f"{label} contains a malformed environment line")
        name, value = line.split("=", 1)
        if not name or name in result:
            raise TimingError(f"{label} contains an empty or duplicate key: {name!r}")
        result[name] = value
    return result, _sha256(raw)


def _validate_env_pair(stock_path: Path, candidate_path: Path) -> tuple[str, str]:
    stock, stock_sha256 = _load_env(stock_path, "stock container environment")
    candidate, candidate_sha256 = _load_env(
        candidate_path, "candidate container environment"
    )
    for label, payload, production in (
        ("stock", stock, "0"),
        ("candidate", candidate, "1"),
    ):
        expected = {
            **_REQUIRED_ENV,
            "FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION": production,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise TimingError(
                    f"{label} container environment {key} mismatch: "
                    f"{payload.get(key)!r} != {value!r}"
                )

    def normalized(payload: dict[str, str]) -> dict[str, str]:
        result = {
            key: value for key, value in payload.items() if key not in _VOLATILE_ENV
        }
        result["FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION"] = "0"
        return result

    if normalized(stock) != normalized(candidate):
        stock_norm = normalized(stock)
        candidate_norm = normalized(candidate)
        differing = sorted(
            key
            for key in stock_norm.keys() | candidate_norm.keys()
            if stock_norm.get(key) != candidate_norm.get(key)
        )
        raise TimingError(
            "container environments differ outside the CFWD selector: "
            + ", ".join(differing)
        )
    return stock_sha256, candidate_sha256


def _validate_eval(path: Path, label: str) -> str:
    report, raw = _load_json(path, f"{label} SWE-Verified eval report")
    if (
        report.get("track") != "swe_bench"
        or report.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or report.get("instance_id") != TASK_ID
        or report.get("verdict") != "resolved"
        or report.get("passed") is not True
        or report.get("harness_exit_code") != 0
    ):
        raise TimingError(f"{label} canonical SWE-Verified task did not resolve")
    return _sha256(raw)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256(raw)


def _validate_census(path: Path, label: str, *, production: bool) -> dict[str, Any]:
    raw = _regular(path, f"{label} fixed32 work census")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(
                line.decode("ascii"),
                object_pairs_hook=_duplicate_checked_object,
                parse_constant=_reject_nonfinite,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, TimingError) as error:
            raise TimingError(
                f"{label} census line {index} is invalid: {error}"
            ) from error
        if not isinstance(record, dict):
            raise TimingError(f"{label} census line {index} is not an object")
        records.append(record)
    if len(records) < 2:
        raise TimingError(f"{label} census lacks complete events and a terminal")
    events = records[:-1]
    terminal = records[-1]
    expected_route = PRODUCTION_ROUTE if production else REFERENCE_ROUTE
    expected_rows = (17, 13) if production else (12, 12)
    expected_launches = 1 if production else 12
    expected_tensor_calls = {
        "full_vocab_softmax_calls": 2 if production else 24,
        "full_vocab_row_gathers": 30 if production else 24,
        "exact_commit_launches": expected_launches,
        "floating_sampling_reimplementation": False,
    }
    for index, event in enumerate(events):
        taw = event.get("taw")
        failures = event.get("failures")
        batch_purity = event.get("batch_purity")
        if (
            event.get("schema") != CENSUS_SCHEMA
            or event.get("event_complete") is not True
            or event.get("event_index") != index
            or event.get("mode") != "hydra27_fixed32"
            or event.get("batch_size") != 1
            or event.get("active_nodes") != 27
            or event.get("physical_drafts") != 31
            or event.get("verify_rows") != 32
            or not isinstance(failures, dict)
            or any(value != 0 for value in failures.values())
            or not isinstance(batch_purity, dict)
            or batch_purity.get("all_physical_31") is not True
            or batch_purity.get("physical_draft_counts") != [31]
            or not isinstance(taw, dict)
            or taw.get("route") != expected_route
            or taw.get("source_contract_schema") != SOURCE_CONTRACT_SCHEMA
            or taw.get("source_contract_sha256") != SOURCE_CONTRACT_SHA256
            or taw.get("table_shape") != [1, 32, 3]
            or taw.get("buffer_capacity") != 32
            or taw.get("loop_iterations") != 12
            or taw.get("uniform_slots") != 36
            or taw.get("vocab_size") != 248_320
            or taw.get("target_rows") != expected_rows[0]
            or taw.get("self_rows") != expected_rows[1]
            or taw.get("exact_commit_launches") != expected_launches
            or taw.get("exact_commit_programs") != expected_launches
        ):
            raise TimingError(
                f"{label} census event {index} violates the fixed32/CFWD contract"
            )
        calls = taw.get("tensor_call_census")
        if not isinstance(calls, dict):
            raise TimingError(f"{label} census event {index} lacks tensor census")
        for key, expected in expected_tensor_calls.items():
            if calls.get(key) != expected:
                raise TimingError(
                    f"{label} census event {index} {key} mismatch: "
                    f"{calls.get(key)!r} != {expected!r}"
                )
    if (
        terminal.get("schema") != CENSUS_TERMINAL_SCHEMA
        or terminal.get("mode") != "hydra27_fixed32"
        or terminal.get("final") is not True
        or terminal.get("event_count") != len(events)
        or terminal.get("first_event_index") != 0
        or terminal.get("last_event_index") != len(events) - 1
        or terminal.get("batch_histogram") != {"1": len(events), "2": 0, "3": 0, "4": 0}
        or terminal.get("events_sha256") != _canonical_sha256(events)
    ):
        raise TimingError(f"{label} census terminal is incomplete or inconsistent")
    return {
        "sha256": _sha256(raw),
        "event_count": len(events),
        "route": expected_route,
        "target_rows_per_event": expected_rows[0],
        "self_rows_per_event": expected_rows[1],
        "full_vocab_softmax_calls_per_event": expected_tensor_calls[
            "full_vocab_softmax_calls"
        ],
        "exact_commit_launches_per_event": expected_launches,
        "fallbacks": 0,
    }


def _require_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise TimingError(f"cannot inspect {label}: {error}") from error
    raise TimingError(f"{label} must be absent from the stock arm")


def _validate_candidate_binding(
    curated_pass: Path,
    stock_selector: Path,
    stock_pass: Path,
    candidate_selector: Path,
    candidate_pass: Path,
) -> dict[str, Any]:
    pass_binding = validate_live_pass(curated_pass)
    _require_absent(stock_selector, "stock CFWD production selector")
    _require_absent(stock_pass, "stock CFWD production credential")
    selector_raw = _regular(candidate_selector, "candidate CFWD selector")
    if selector_raw != b"1\n":
        raise TimingError(
            "candidate CFWD production selector does not contain exactly 1"
        )
    copied_raw = _regular(candidate_pass, "candidate copied CFWD credential")
    curated_raw = _regular(curated_pass, "curated CFWD live PASS")
    if copied_raw != curated_raw or _sha256(copied_raw) != LIVE_PASS_SHA256:
        raise TimingError("candidate copied credential differs from the curated PASS")
    return {
        **pass_binding,
        "selector_sidecar_sha256": _sha256(selector_raw),
        "copied_credential_sha256": _sha256(copied_raw),
    }


def reduce_pair(
    *,
    subset: Path,
    curated_pass: Path,
    stock_measure: Path,
    candidate_measure: Path,
    stock_container_env: Path,
    candidate_container_env: Path,
    stock_census: Path,
    candidate_census: Path,
    stock_eval_report: Path,
    candidate_eval_report: Path,
    stock_selector: Path,
    stock_production_pass: Path,
    candidate_selector: Path,
    candidate_production_pass: Path,
    source_commit: str,
) -> dict[str, Any]:
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise TimingError("timing source commit is invalid")
    subset_payload, subset_raw = _load_json(subset, "canonical B1 subset")
    if _sha256(subset_raw) != SUBSET_SHA256:
        raise TimingError("canonical B1 subset SHA-256 drifted")
    if subset_payload.get("instance_ids") != [TASK_ID]:
        raise TimingError("canonical B1 subset task ID drifted")

    stock, _ = _load_json(stock_measure, "stock full-wall measurement")
    candidate, _ = _load_json(candidate_measure, "candidate full-wall measurement")
    stock_values = _validate_measure(stock, "stock")
    candidate_values = _validate_measure(candidate, "candidate")
    stock_env_sha256, candidate_env_sha256 = _validate_env_pair(
        stock_container_env, candidate_container_env
    )
    stock_eval_sha256 = _validate_eval(stock_eval_report, "stock")
    candidate_eval_sha256 = _validate_eval(candidate_eval_report, "candidate")
    stock_work = _validate_census(stock_census, "stock", production=False)
    candidate_work = _validate_census(candidate_census, "candidate", production=True)
    binding = _validate_candidate_binding(
        curated_pass,
        stock_selector,
        stock_production_pass,
        candidate_selector,
        candidate_production_pass,
    )

    def arm(
        selector: str,
        values: dict[str, float],
        env_sha256: str,
        eval_sha256: str,
        work: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "selector": selector,
            "cfwd_gpu_ms_per_event": values["cfwd_gpu_ms_per_event"],
            "full_wall_ms_per_event": values["step_wall_ms"],
            "measured_tps_fullstep_wall": values["measured_tps_fullstep_wall"],
            "accepted_drafts_per_event": values["accept_per_event"],
            "committed_tokens_per_event": values["committed_per_event"],
            "s_fwd_gpu_ms_per_event": values["s_fwd_gpu_ms_per_event"],
            "drafter_gpu_ms_per_event": values["drafter_gpu_ms_per_event"],
            "wall_steps_measured": values["wall_steps_measured"],
            "step_wall_to_mandatory_weight_floor_ratio": values["floor_ratio"],
            "container_env_sha256": env_sha256,
            "eval_report_sha256": eval_sha256,
            "task_verdict": "resolved",
            "fixed_work_census": work,
        }

    stock_arm = arm(
        "fixed32_reference_commit_walk",
        stock_values,
        stock_env_sha256,
        stock_eval_sha256,
        stock_work,
    )
    candidate_arm = {
        **arm(
            CANDIDATE,
            candidate_values,
            candidate_env_sha256,
            candidate_eval_sha256,
            candidate_work,
        ),
        "production_engagement": binding,
    }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "run_classification": (
            "one_real_swe_verified_b1_cfwd_all_parent_timing_diagnostic"
        ),
        "task_count": 1,
        "task_ids": [TASK_ID],
        "batch_size": 1,
        "concurrency": 1,
        "source_commit": source_commit,
        "candidate_base_commit": CANDIDATE_BASE_COMMIT,
        "topology": "hydra27_fixed32",
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "common_physical_work": {
            "physical_drafts_per_event": 31,
            "physical_rows_per_event": 32,
            "walk_cap": 12,
            "full_vocabulary_rows": 248_320,
            "identical_in_both_arms": True,
        },
        "only_arm_delta": ("FR13_FIXED32_TAW_NATIVE_PRECOMPUTE_PRODUCTION=0 to 1"),
        "decision_metrics": [
            "cfwd_gpu_ms_per_event",
            "full_wall_ms_per_event",
            "measured_tps_fullstep_wall",
            "accepted_drafts_per_event",
        ],
        "stock_reference": stock_arm,
        "candidate": candidate_arm,
        "candidate_minus_stock": {
            "cfwd_gpu_ms_per_event": (
                candidate_values["cfwd_gpu_ms_per_event"]
                - stock_values["cfwd_gpu_ms_per_event"]
            ),
            "full_wall_ms_per_event": (
                candidate_values["step_wall_ms"] - stock_values["step_wall_ms"]
            ),
            "accepted_drafts_per_event": (
                candidate_values["accept_per_event"] - stock_values["accept_per_event"]
            ),
        },
        "candidate_to_stock_full_wall_tps_ratio": (
            candidate_values["measured_tps_fullstep_wall"]
            / stock_values["measured_tps_fullstep_wall"]
        ),
        "stock_to_candidate_cfwd_ratio": (
            stock_values["cfwd_gpu_ms_per_event"]
            / candidate_values["cfwd_gpu_ms_per_event"]
        ),
        "stock_to_candidate_full_wall_ratio": (
            stock_values["step_wall_ms"] / candidate_values["step_wall_ms"]
        ),
        "floor_contract": {
            "mandatory_weight_bytes": FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
            "mandatory_weight_floor_ms_per_step": (
                FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
            ),
            "one_sided_1_15x_cap_ms_per_step": FULL_VOCAB_CAP_MS,
            "mandatory_weight_floor_is_complete_step_floor": False,
        },
        "timing_eligible": False,
        "timing_ineligible_reason": (
            "one-task B1 diagnostic; standing acceptance requires an eligible "
            "real task set and the formal statistical procedure"
        ),
        "floor_acceptance_eligible": False,
        "formal_floor_acceptance_eligible": False,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pass_check = commands.add_parser("validate-pass")
    pass_check.add_argument("--path", required=True, type=Path)
    eval_check = commands.add_parser("validate-eval")
    eval_check.add_argument("--path", required=True, type=Path)
    eval_check.add_argument("--label", default="timing arm")

    reduce_parser = commands.add_parser("reduce")
    for name in (
        "subset",
        "curated-pass",
        "stock-measure",
        "candidate-measure",
        "stock-container-env",
        "candidate-container-env",
        "stock-census",
        "candidate-census",
        "stock-eval-report",
        "candidate-eval-report",
        "stock-selector",
        "stock-production-pass",
        "candidate-selector",
        "candidate-production-pass",
    ):
        reduce_parser.add_argument(f"--{name}", required=True, type=Path)
    reduce_parser.add_argument("--source-commit", required=True)
    reduce_parser.add_argument("--out", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "validate-pass":
            payload = validate_live_pass(args.path)
            print(json.dumps(payload, sort_keys=True))
            return 0
        if args.command == "validate-eval":
            print(_validate_eval(args.path, args.label))
            return 0
        payload = reduce_pair(
            subset=args.subset,
            curated_pass=args.curated_pass,
            stock_measure=args.stock_measure,
            candidate_measure=args.candidate_measure,
            stock_container_env=args.stock_container_env,
            candidate_container_env=args.candidate_container_env,
            stock_census=args.stock_census,
            candidate_census=args.candidate_census,
            stock_eval_report=args.stock_eval_report,
            candidate_eval_report=args.candidate_eval_report,
            stock_selector=args.stock_selector,
            stock_production_pass=args.stock_production_pass,
            candidate_selector=args.candidate_selector,
            candidate_production_pass=args.candidate_production_pass,
            source_commit=args.source_commit,
        )
        _write(args.out, payload)
        return 0
    except TimingError as error:
        print(f"FAIL CFWD all-parent timing: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
