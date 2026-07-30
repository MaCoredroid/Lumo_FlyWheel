#!/usr/bin/env python3
"""Reduce a fixed32 Nsight Systems report to privacy-safe GPU attribution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from math import ceil
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if os.fspath(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, os.fspath(SCRIPT_DIR))
import fr13_fixed32_contract as fixed32_contract  # noqa: E402


DEFAULT_NSYS_BIN = Path("/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys")
REPORT_NAMES = (
    "nvtx_gpu_proj_sum",
    "nvtx_kern_sum",
    "cuda_gpu_kern_sum",
)
STEP_RANGE = "fr13.fixed32.step"
PHASE_RANGES = {
    "sfwd": "fr13.fixed32.sfwd",
    "postprocess": "fr13.fixed32.postprocess",
    "cfwd": "fr13.fixed32.cfwd",
    "dfwd": "fr13.fixed32.dfwd",
}
ATTRIBUTION_RANGES = {"step": STEP_RANGE, **PHASE_RANGES}
FIXED32_RANGE_PREFIX = "fr13.fixed32."
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class ReductionError(RuntimeError):
    """The report cannot produce complete fixed32 attribution."""


def _normalize_header(value: str) -> str:
    value = value.lstrip("\ufeff").strip()
    value = re.sub(r"\s*(?:\([^)]*\)|\[[^]]*])\s*$", "", value)
    value = value.split(":", maxsplit=1)[0]
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _parse_stats_csv(
    text: str,
    *,
    report_name: str,
    required_columns: Sequence[str],
) -> list[dict[str, str]]:
    """Find and parse a CSV table after any nsys NOTICE/status prelude."""

    lines = text.splitlines()
    required = {_normalize_header(column) for column in required_columns}
    header_index: int | None = None

    for index, line in enumerate(lines):
        try:
            candidate = next(csv.reader([line]))
        except csv.Error:
            continue
        normalized = {_normalize_header(column) for column in candidate}
        if required <= normalized:
            header_index = index
            break

    if header_index is None:
        expected = ", ".join(required_columns)
        raise ReductionError(
            f"{report_name} CSV header is missing required columns: {expected}"
        )

    reader = csv.reader(io.StringIO("\n".join(lines[header_index:])))
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise ReductionError(f"{report_name} CSV table is empty") from exc

    header = [_normalize_header(column) for column in raw_header]
    if len(set(header)) != len(header):
        raise ReductionError(f"{report_name} CSV has duplicate normalized columns")

    rows: list[dict[str, str]] = []
    for row_number, row in enumerate(reader, start=2):
        if not row or all(not value.strip() for value in row):
            continue
        if len(row) != len(header):
            raise ReductionError(
                f"{report_name} CSV row {row_number} has "
                f"{len(row)} values for {len(header)} columns"
            )
        rows.append(dict(zip(header, row, strict=True)))

    if not rows:
        raise ReductionError(f"{report_name} CSV contains no data rows")
    return rows


def _optional_field(row: Mapping[str, str], column: str) -> str | None:
    key = _normalize_header(column)
    try:
        value = row[key].strip()
    except KeyError as exc:
        raise ReductionError(f"parsed CSV is missing column {column!r}") from exc
    return value or None


def _field(row: Mapping[str, str], column: str) -> str:
    value = _optional_field(row, column)
    if value is None:
        raise ReductionError(f"parsed CSV has an empty {column!r} value")
    return value


def _nonnegative_integer(row: Mapping[str, str], column: str) -> int:
    raw = _field(row, column).replace(",", "")
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ReductionError(f"parsed CSV has a non-numeric {column!r} value") from exc
    if not value.is_finite() or value < 0 or value != value.to_integral_value():
        raise ReductionError(f"parsed CSV has a non-integer {column!r} value")
    return int(value)


def _require_exact_phase_ranges(
    rows: Sequence[Mapping[str, str]],
    *,
    range_column: str,
    report_name: str,
) -> None:
    expected = set(ATTRIBUTION_RANGES.values())
    observed = {
        value
        for row in rows
        if (value := _optional_field(row, range_column)) is not None
        and value.startswith(FIXED32_RANGE_PREFIX)
    }
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ReductionError(
            f"{report_name} fixed32 NVTX ranges do not match exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _projection_by_range(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, int | str]]:
    _require_exact_phase_ranges(
        rows,
        range_column="Range",
        report_name="nvtx_gpu_proj_sum",
    )
    result: dict[str, dict[str, int | str]] = {}
    for phase, nvtx_range in ATTRIBUTION_RANGES.items():
        matches = [row for row in rows if _optional_field(row, "Range") == nvtx_range]
        if len(matches) != 1:
            raise ReductionError(
                "nvtx_gpu_proj_sum must contain exactly one row for "
                f"{nvtx_range}; found {len(matches)}"
            )
        row = matches[0]
        result[phase] = {
            "nvtx_range": nvtx_range,
            "projected_gpu_time_ns": _nonnegative_integer(row, "Total Proj Time"),
            "range_instances": _nonnegative_integer(row, "Range Instances"),
            "gpu_ops": _nonnegative_integer(row, "Total GPU Ops"),
        }
    return result


def _range_top_kernels(
    rows: Sequence[Mapping[str, str]],
    *,
    top: int,
) -> dict[str, list[dict[str, int | str]]]:
    _require_exact_phase_ranges(
        rows,
        range_column="NVTX Range",
        report_name="nvtx_kern_sum",
    )
    result: dict[str, list[dict[str, int | str]]] = {}
    for phase, nvtx_range in ATTRIBUTION_RANGES.items():
        aggregate: dict[str, dict[str, int | str]] = {}
        for row in rows:
            if _optional_field(row, "NVTX Range") != nvtx_range:
                continue
            name = _field(row, "Kernel Name")
            kernel = aggregate.setdefault(
                name,
                {
                    "name": name,
                    "total_time_ns": 0,
                    "instances": 0,
                    "nvtx_instances": 0,
                },
            )
            kernel["total_time_ns"] = int(kernel["total_time_ns"]) + (
                _nonnegative_integer(row, "Total Time")
            )
            kernel["instances"] = int(kernel["instances"]) + (
                _nonnegative_integer(row, "Kern Inst")
            )
            kernel["nvtx_instances"] = int(kernel["nvtx_instances"]) + (
                _nonnegative_integer(row, "NVTX Inst")
            )
        if not aggregate:
            raise ReductionError(f"nvtx_kern_sum has no kernels for {nvtx_range}")
        result[phase] = sorted(
            aggregate.values(),
            key=lambda item: (-int(item["total_time_ns"]), str(item["name"])),
        )[:top]
    return result


def _overall_top_kernels(
    rows: Sequence[Mapping[str, str]],
    *,
    top: int,
) -> list[dict[str, int | str]]:
    aggregate: dict[str, dict[str, int | str]] = {}
    for row in rows:
        name = _field(row, "Name")
        kernel = aggregate.setdefault(
            name,
            {
                "name": name,
                "total_time_ns": 0,
                "instances": 0,
            },
        )
        kernel["total_time_ns"] = int(kernel["total_time_ns"]) + (
            _nonnegative_integer(row, "Total Time")
        )
        kernel["instances"] = int(kernel["instances"]) + (
            _nonnegative_integer(row, "Instances")
        )
    return sorted(
        aggregate.values(),
        key=lambda item: (-int(item["total_time_ns"]), str(item["name"])),
    )[:top]


def _build_summary(
    *,
    report_sha256: str,
    report_bytes: int,
    stats_csv: Mapping[str, str],
    top: int,
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", report_sha256):
        raise ReductionError("report SHA-256 must be lowercase hexadecimal")
    if report_bytes <= 0:
        raise ReductionError("Nsight report must be nonempty")
    if top <= 0:
        raise ReductionError("--top must be positive")

    projection_rows = _parse_stats_csv(
        stats_csv["nvtx_gpu_proj_sum"],
        report_name="nvtx_gpu_proj_sum",
        required_columns=(
            "Range",
            "Total Proj Time",
            "Range Instances",
            "Total GPU Ops",
        ),
    )
    phase_kernel_rows = _parse_stats_csv(
        stats_csv["nvtx_kern_sum"],
        report_name="nvtx_kern_sum",
        required_columns=(
            "NVTX Range",
            "NVTX Inst",
            "Kern Inst",
            "Total Time",
            "Kernel Name",
        ),
    )
    overall_kernel_rows = _parse_stats_csv(
        stats_csv["cuda_gpu_kern_sum"],
        report_name="cuda_gpu_kern_sum",
        required_columns=("Total Time", "Instances", "Name"),
    )

    ranges = _projection_by_range(projection_rows)
    range_kernels = _range_top_kernels(phase_kernel_rows, top=top)
    for phase in ranges:
        ranges[phase]["top_kernels"] = range_kernels[phase]

    step = ranges.pop("step")
    phases = ranges
    child_projected_ns = sum(
        int(phase["projected_gpu_time_ns"]) for phase in phases.values()
    )
    step_projected_ns = int(step.pop("projected_gpu_time_ns"))
    step_instances = int(step["range_instances"])
    if step_instances <= 0:
        raise ReductionError("step envelope contains no completed NVTX ranges")
    child_instance_deltas = {
        phase: int(values["range_instances"]) - step_instances
        for phase, values in phases.items()
    }
    if any(abs(delta) > 1 for delta in child_instance_deltas.values()):
        raise ReductionError(
            "fixed32 child/step NVTX instance counts differ by more than "
            f"one capture boundary: {child_instance_deltas}"
        )
    boundary_allowance_ns = sum(
        ceil(
            int(phases[phase]["projected_gpu_time_ns"])
            / int(phases[phase]["range_instances"])
        )
        for phase, delta in child_instance_deltas.items()
        if delta == 1 and int(phases[phase]["range_instances"]) > 0
    )
    signed_residual_ns = step_projected_ns - child_projected_ns
    reconciliation_tolerance_ns = max(1_000, step_projected_ns // 10_000)
    if signed_residual_ns < -(reconciliation_tolerance_ns + boundary_allowance_ns):
        raise ReductionError(
            "disjoint fixed32 child projections exceed the step envelope: "
            f"step={step_projected_ns}, children={child_projected_ns}, "
            f"rounding_tolerance={reconciliation_tolerance_ns}, "
            f"boundary_allowance={boundary_allowance_ns}"
        )
    step["step_projected_gpu_time_ns"] = step_projected_ns
    step["child_projected_gpu_time_ns"] = child_projected_ns
    step["residual_projected_gpu_time_ns"] = signed_residual_ns
    step["child_instance_delta_from_step"] = child_instance_deltas
    step["capture_boundary_allowance_ns"] = boundary_allowance_ns
    step["negative_residual_tolerance_ns"] = reconciliation_tolerance_ns

    return {
        "acceptance_valid": False,
        "attribution_only": True,
        "curated_publishable": False,
        "overall_top_kernels": _overall_top_kernels(overall_kernel_rows, top=top),
        "phases": phases,
        "projection_semantics": {
            "children_are_disjoint": True,
            "step_is_envelope_not_additive_phase": True,
            "time_basis": "first_to_last_projected_gpu_operation",
        },
        "provenance_bound": False,
        "raw_profiler_artifacts_publishable": False,
        "report": {
            "bytes": report_bytes,
            "sha256": report_sha256,
        },
        "schema": "fr13.fixed32.nsys_attribution.v2",
        "step_envelope": step,
    }


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key!r}")
        payload[key] = value
    return payload


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _strict_json_value(raw: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReductionError(f"{label} is not strict UTF-8 JSON: {exc}") from exc


def _strict_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReductionError(f"{label} is unavailable") from exc
    if not raw:
        raise ReductionError(f"{label} is empty")
    payload = _strict_json_value(raw, label=label)
    if not isinstance(payload, dict):
        raise ReductionError(f"{label} must be a JSON object")
    return payload, raw


def _strict_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ReductionError(f"{label} is unavailable") from exc
    if not raw or not raw.endswith(b"\n"):
        raise ReductionError(f"{label} is empty or truncated")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        payload = _strict_json_value(line, label=f"{label} row {index}")
        if not isinstance(payload, dict):
            raise ReductionError(f"{label} row {index} must be an object")
        rows.append(payload)
    return rows, raw


def _exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        unexpected = sorted(set(payload) - expected)
        raise ReductionError(
            f"{label} keys mismatch; missing={missing}, unexpected={unexpected}"
        )


def _artifact_identity(raw: bytes) -> dict[str, int | str]:
    return {
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _require_exact_artifact_path(
    path: Path,
    expected_path: Path,
    *,
    label: str,
) -> None:
    if path.is_symlink():
        raise ReductionError(f"{label} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ReductionError(f"{label} is unavailable") from exc
    if resolved != expected_path.resolve():
        raise ReductionError(f"{label} is not the expected arm-local artifact")


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ReductionError(f"{label} must be an array of strings")
    return value


def _environment_map(value: Any, *, label: str) -> dict[str, str]:
    entries = _string_list(value, label=label)
    environment: dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ReductionError(f"{label} contains a malformed entry")
        name, setting = entry.split("=", 1)
        if not name or name in environment:
            raise ReductionError(f"{label} contains an empty or duplicate name")
        environment[name] = setting
    return environment


def _validate_process_identity(
    path: Path,
    *,
    arm_dir: Path,
    concurrency: int,
) -> dict[str, Any]:
    expected_path = arm_dir / "fixed32_process_identity.json"
    _require_exact_artifact_path(path, expected_path, label="process identity")
    payload, raw = _strict_json_object(path, label="process identity")
    _exact_keys(
        payload,
        {"schema", "pid1", "engine_core"},
        label="process identity",
    )
    if payload["schema"] != "fr13-fixed32-process-identity-v1":
        raise ReductionError("process identity schema mismatch")

    pid1 = payload["pid1"]
    engine = payload["engine_core"]
    record_keys = {"pid", "argv", "environ", "forked_fa2_maps"}
    if not isinstance(pid1, dict) or not isinstance(engine, dict):
        raise ReductionError("process identity records must be objects")
    _exact_keys(pid1, record_keys, label="PID1 process identity")
    _exact_keys(engine, record_keys, label="EngineCore process identity")
    if isinstance(pid1["pid"], bool) or pid1["pid"] != 1:
        raise ReductionError("process identity PID1 record is invalid")
    if (
        isinstance(engine["pid"], bool)
        or not isinstance(engine["pid"], int)
        or engine["pid"] <= 1
    ):
        raise ReductionError("process identity EngineCore PID is invalid")

    pid1_argv = _string_list(pid1["argv"], label="PID1 argv")
    try:
        fixed32_contract.validate_process_pid1_argv(
            pid1_argv,
            concurrency,
            attribution_only=True,
        )
    except fixed32_contract.ContractError as exc:
        raise ReductionError(
            "process identity PID1 argv is not the exact attribution contract"
        ) from exc

    pid1_environment = _environment_map(
        pid1["environ"],
        label="PID1 environment",
    )
    expected_profile_environment = {
        "FR13_FIXED32_ATTRIBUTION_ONLY": "1",
        "FR13_FIXED32_NVTX_PROFILE": "1",
        "LUMO_NSYS_WRAP_VLLM": "1",
    }
    if any(
        pid1_environment.get(name) != setting
        for name, setting in expected_profile_environment.items()
    ):
        raise ReductionError(
            "process identity PID1 environment is not attribution-only"
        )
    _string_list(pid1["forked_fa2_maps"], label="PID1 FA2 maps")

    if _string_list(engine["argv"], label="EngineCore argv") != [
        "VLLM::EngineCore"
    ]:
        raise ReductionError("process identity EngineCore argv is not exact")
    _environment_map(engine["environ"], label="EngineCore environment")
    engine_maps = _string_list(
        engine["forked_fa2_maps"],
        label="EngineCore FA2 maps",
    )
    pinned_fa2 = str(fixed32_contract.CONTAINER_FA2_DESTINATION)
    if not any(pinned_fa2 in mapping for mapping in engine_maps):
        raise ReductionError(
            "process identity does not prove the pinned EngineCore FA2 mapping"
        )

    return {
        **_artifact_identity(raw),
        "engine_core_identity_exact": True,
        "engine_core_pinned_fa2_mapped": True,
        "pid1_attribution_contract_exact": True,
        "profile_environment_contract_exact": True,
        "schema": "fr13-fixed32-process-identity-v1",
    }


def _validate_container_identity(
    path: Path,
    *,
    arm_dir: Path,
) -> dict[str, Any]:
    expected_path = arm_dir / "fixed32_container_identity.json"
    _require_exact_artifact_path(path, expected_path, label="container identity")
    payload, raw = _strict_json_object(path, label="container identity")
    _exact_keys(
        payload,
        {
            "schema",
            "name",
            "image_id",
            "configured_image",
            "platform",
            "running",
        },
        label="container identity",
    )
    expected = {
        "schema": "fr13-fixed32-container-identity-v1",
        "name": f"/fr13-bigdenom-{arm_dir.name}",
        "image_id": fixed32_contract.IMAGE_ID,
        "configured_image": fixed32_contract.IMAGE_REFERENCE,
        "platform": fixed32_contract.IMAGE_OS,
        "running": True,
    }
    if payload != expected:
        raise ReductionError("container identity does not match the pinned contract")
    return {
        **_artifact_identity(raw),
        "arm_name_bound": True,
        "pinned_container_contract_exact": True,
        "running_at_identity_capture": True,
        "schema": "fr13-fixed32-container-identity-v1",
    }


def _validate_runtime_attestation(
    path: Path,
    *,
    arm_dir: Path,
) -> dict[str, Any]:
    expected_path = arm_dir / "logs" / "fr13_fixed32_runtime_attestation.json"
    _require_exact_artifact_path(path, expected_path, label="runtime attestation")
    payload, raw = _strict_json_object(path, label="runtime attestation")
    _exact_keys(
        payload,
        {
            "schema",
            "canonical_format",
            "python",
            "vllm",
            "forked_fa2",
            "arctic",
            "overall_canonical_sha256",
        },
        label="runtime attestation",
    )
    try:
        validated = fixed32_contract.validate_runtime_attestation(payload)
    except fixed32_contract.ContractError as exc:
        raise ReductionError(
            "runtime attestation does not match the pinned contract"
        ) from exc
    canonical_sha256 = _require_digest(
        validated.get("overall_canonical_sha256"),
        label="runtime attestation canonical SHA-256",
    )
    return {
        **_artifact_identity(raw),
        "overall_canonical_sha256": canonical_sha256,
        "pinned_runtime_contract_exact": True,
        "schema": fixed32_contract.RUNTIME_SCHEMA,
    }


def _validate_exact4_subset(path: Path) -> dict[str, Any]:
    payload, raw = _strict_json_object(path, label="canonical exact4 subset")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXACT4_SUBSET_SHA256:
        raise ReductionError("canonical exact4 subset SHA-256 drift")
    if (
        payload.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or payload.get("split") != "test"
        or payload.get("instance_ids") != list(EXACT4_TASK_IDS)
    ):
        raise ReductionError("canonical exact4 SWE-Verified identity mismatch")
    return {
        **_artifact_identity(raw),
        "dataset": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "task_count": len(EXACT4_TASK_IDS),
    }


def _validate_manifest_pair(
    launch_path: Path,
    end_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    _launch, launch_raw = _strict_json_object(
        launch_path, label=f"{label} launch manifest"
    )
    _end, end_raw = _strict_json_object(end_path, label=f"{label} end manifest")
    if launch_raw != end_raw:
        raise ReductionError(f"{label} launch/end manifests are not byte-equal")
    return {
        **_artifact_identity(launch_raw),
        "launch_end_byte_equal": True,
    }


def _nonbool_zero(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def _require_digest(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ReductionError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _validate_pretask_zero_traffic(
    path: Path,
    *,
    mode: str,
) -> dict[str, Any]:
    marker, marker_raw = _strict_json_object(path, label="pretask zero-traffic marker")
    _exact_keys(
        marker,
        {
            "schema",
            "mode",
            "no_positive_probe",
            "generation_probe_commands_executed",
            "metrics",
            "work_census",
            "ready_ack",
        },
        label="pretask zero-traffic marker",
    )
    if (
        marker["schema"] != "fr13-fixed32-pretask-zero-traffic-v1"
        or marker["mode"] != mode
        or marker["no_positive_probe"] is not True
        or not _nonbool_zero(marker["generation_probe_commands_executed"])
    ):
        raise ReductionError("pretask zero-traffic claim is invalid")

    arm_dir = path.parent.resolve()
    metrics = marker["metrics"]
    if not isinstance(metrics, dict):
        raise ReductionError("pretask metrics identity must be an object")
    _exact_keys(
        metrics,
        {"path", "sha256", "spec_drafts", "spec_tokens"},
        label="pretask metrics identity",
    )
    metrics_path = arm_dir / "metrics_before_swe.txt"
    metrics_sha256, metrics_bytes = _sha256_file(metrics_path)
    if (
        metrics.get("path") != str(metrics_path.resolve())
        or _require_digest(metrics.get("sha256"), label="pretask metrics SHA-256")
        != metrics_sha256
        or not _nonbool_zero(metrics.get("spec_drafts"))
        or not _nonbool_zero(metrics.get("spec_tokens"))
    ):
        raise ReductionError("pretask metrics identity is not exact zero")

    census = marker["work_census"]
    if not isinstance(census, dict):
        raise ReductionError("pretask work-census identity must be an object")
    _exact_keys(
        census,
        {"path", "exists", "bytes", "sha256"},
        label="pretask work-census identity",
    )
    census_path = arm_dir / "logs" / "fr13_fixed32_work_census.jsonl"
    if (
        census.get("path") != str(census_path.resolve())
        or not isinstance(census.get("exists"), bool)
        or not _nonbool_zero(census.get("bytes"))
        or census.get("sha256") != EMPTY_SHA256
    ):
        raise ReductionError("pretask work census was not empty")

    ready = marker["ready_ack"]
    if not isinstance(ready, dict):
        raise ReductionError("pretask ready-ack identity must be an object")
    _exact_keys(
        ready,
        {"path", "sha256", "generation"},
        label="pretask ready-ack identity",
    )
    ready_path = arm_dir / "fixed32_ready_ack.json"
    ready_sha256, ready_bytes = _sha256_file(ready_path)
    if (
        ready.get("path") != str(ready_path.resolve())
        or _require_digest(ready.get("sha256"), label="pretask ready-ack SHA-256")
        != ready_sha256
        or not _nonbool_zero(ready.get("generation"))
    ):
        raise ReductionError("pretask ready-ack identity is invalid")

    return {
        **_artifact_identity(marker_raw),
        "generation_probe_commands_executed": 0,
        "metrics": {
            "bytes": metrics_bytes,
            "sha256": metrics_sha256,
            "spec_drafts": 0,
            "spec_tokens": 0,
        },
        "no_positive_probe": True,
        "ready_ack": {
            "bytes": ready_bytes,
            "generation": 0,
            "sha256": ready_sha256,
        },
        "work_census_at_task_start": {
            "bytes": 0,
            "sha256": EMPTY_SHA256,
        },
    }


def _digest_values(values: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(values),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _validate_ingress_ledgers(
    proxy_path: Path,
    engine_path: Path,
) -> dict[str, Any]:
    try:
        from lumo_flywheel_serving.inference_proxy import (
            Fixed32IngressError,
            fixed32_canonical_task_set_sha256,
            fixed32_task_key_id,
            verify_fixed32_ingress_ledger,
        )
    except ImportError as exc:
        raise ReductionError("fixed32 ingress verifier is unavailable") from exc

    expected_task_keys = {fixed32_task_key_id(task_id) for task_id in EXACT4_TASK_IDS}
    canonical_task_set_sha256 = fixed32_canonical_task_set_sha256(EXACT4_TASK_IDS)
    proxy_rows, proxy_raw = _strict_jsonl(proxy_path, label="proxy ingress ledger")
    engine_rows, engine_raw = _strict_jsonl(engine_path, label="engine ingress ledger")
    try:
        proxy_verification = verify_fixed32_ingress_ledger(
            proxy_path,
            expected_role="proxy",
            require_finalized=False,
        )
        engine_verification = verify_fixed32_ingress_ledger(
            engine_path,
            expected_role="engine",
            require_finalized=False,
        )
    except Fixed32IngressError as exc:
        raise ReductionError(f"fixed32 ingress ledger is invalid: {exc}") from exc

    for role, rows in (("proxy", proxy_rows), ("engine", engine_rows)):
        begins = [row for row in rows if row.get("event") == "campaign_begin"]
        if (
            len(begins) != 1
            or begins[0].get("evidence_sha256") != canonical_task_set_sha256
        ):
            raise ReductionError(f"{role} ledger is not bound to the exact4 task set")
        observed_task_keys = {
            value for row in rows if (value := row.get("task_key_id")) is not None
        }
        if not observed_task_keys <= expected_task_keys:
            raise ReductionError(f"{role} ledger contains a noncanonical task key")

    completed_logical_keys = {
        str(row["task_key_id"])
        for row in proxy_rows
        if row.get("event") == "logical_complete" and row.get("outcome") == "completed"
    }
    proxy_completed_attempts = {
        (
            str(row["task_key_id"]),
            str(row["engine_request_id_sha256"]),
            str(row["evidence_sha256"]),
        )
        for row in proxy_rows
        if row.get("event") == "attempt_result"
        and row.get("outcome") == "response"
        and isinstance(row.get("status_code"), int)
        and not isinstance(row.get("status_code"), bool)
        and 200 <= int(row["status_code"]) < 300
    }
    engine_completed_attempts = {
        (
            str(row["task_key_id"]),
            str(row["engine_request_id_sha256"]),
            str(row["evidence_sha256"]),
        )
        for row in engine_rows
        if row.get("event") == "request_complete" and row.get("outcome") == "completed"
    }
    matched_attempts = {
        attempt
        for attempt in proxy_completed_attempts & engine_completed_attempts
        if attempt[0] in completed_logical_keys
    }
    if not matched_attempts:
        raise ReductionError(
            "ingress ledgers contain no cross-bound completed real-SWE request"
        )
    matched_task_keys = sorted({attempt[0] for attempt in matched_attempts})

    return {
        "canonical_task_key_set_sha256": _digest_values(sorted(expected_task_keys)),
        "canonical_task_set_sha256": canonical_task_set_sha256,
        "engine": {
            **_artifact_identity(engine_raw),
            "completed_requests": len(engine_completed_attempts),
            "verification": engine_verification,
        },
        "matched_completed_attempts": len(matched_attempts),
        "matched_completed_task_count": len(matched_task_keys),
        "matched_task_key_set_sha256": _digest_values(matched_task_keys),
        "proxy": {
            **_artifact_identity(proxy_raw),
            "completed_logical_requests": len(completed_logical_keys),
            "successful_attempts": len(proxy_completed_attempts),
            "verification": proxy_verification,
        },
    }


def _nsys_version(nsys_bin: Path) -> str:
    completed = subprocess.run(
        [os.fspath(nsys_bin), "--version"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "LC_ALL": "C"},
    )
    lines = [
        line.strip()
        for line in (completed.stdout + completed.stderr).splitlines()
        if line.strip()
    ]
    if (
        completed.returncode != 0
        or len(lines) != 1
        or not lines[0].startswith("NVIDIA Nsight Systems version ")
        or len(lines[0]) > 200
    ):
        raise ReductionError("could not bind an exact Nsight Systems version")
    return lines[0]


def _build_attribution_provenance(
    *,
    subset: Path,
    runtime_manifest_launch: Path,
    runtime_manifest_end: Path,
    external_manifest_launch: Path,
    external_manifest_end: Path,
    process_identity: Path,
    container_identity: Path,
    runtime_attestation: Path,
    pretask_zero_traffic: Path,
    proxy_ledger: Path,
    engine_ledger: Path,
    mode: str,
    batch_size: int,
    concurrency: int,
    driver_rc: int,
    nsys_delay_s: int,
    nsys_duration_s: int,
    nsys_flush_ms: int,
    nsys_trace: str,
    nsys_config_directives: str,
    nsys_discard_environment: bool,
    nsys_bin: Path,
) -> dict[str, Any]:
    if mode != "tail6_fixed32":
        raise ReductionError("B1 profiler mode must be tail6_fixed32")
    if batch_size != 1 or concurrency != 1:
        raise ReductionError("attribution provenance requires real-SWE B1")
    if isinstance(driver_rc, bool) or not 1 <= driver_rc <= 255:
        raise ReductionError(
            "bounded attribution driver must have a nonzero shell exit code"
        )
    if nsys_delay_s != 1200 or nsys_duration_s != 300:
        raise ReductionError(
            "Nsight capture must use the canonical 1200s delay/300s duration"
        )
    if nsys_flush_ms != 100:
        raise ReductionError("Nsight attribution requires a 100ms CUDA flush")
    if nsys_trace != "cuda,cuda-sw,nvtx":
        raise ReductionError("Nsight attribution requires cuda,cuda-sw,nvtx tracing")
    if nsys_config_directives != "CuptiUseRawGpuTimestamps=false":
        raise ReductionError("Nsight GPU timestamp directive drift")
    if nsys_discard_environment is not True:
        raise ReductionError("Nsight report environment capture must be disabled")

    subset_identity = _validate_exact4_subset(subset)
    pretask_identity = _validate_pretask_zero_traffic(
        pretask_zero_traffic,
        mode=mode,
    )
    arm_dir = pretask_zero_traffic.parent.resolve()
    return {
        "batch_size": 1,
        "bounded_capture": True,
        "container_identity": _validate_container_identity(
            container_identity,
            arm_dir=arm_dir,
        ),
        "concurrency": 1,
        "driver_exit_code": driver_rc,
        "exact4_subset": subset_identity,
        "external_manifest": _validate_manifest_pair(
            external_manifest_launch,
            external_manifest_end,
            label="external",
        ),
        "ingress": _validate_ingress_ledgers(proxy_ledger, engine_ledger),
        "mode": mode,
        "nsight": {
            "config_directives": nsys_config_directives,
            "delay_is_process_relative": True,
            "delay_s": nsys_delay_s,
            "discard_environment": True,
            "duration_s": nsys_duration_s,
            "flush_ms": nsys_flush_ms,
            "trace": nsys_trace,
            "version": _nsys_version(nsys_bin),
        },
        "pretask_zero_traffic": pretask_identity,
        "process_identity": _validate_process_identity(
            process_identity,
            arm_dir=arm_dir,
            concurrency=concurrency,
        ),
        "real_swe_verified": True,
        "runtime_attestation": _validate_runtime_attestation(
            runtime_attestation,
            arm_dir=arm_dir,
        ),
        "runtime_manifest": _validate_manifest_pair(
            runtime_manifest_launch,
            runtime_manifest_end,
            label="runtime",
        ),
        "schema": "fr13.fixed32.nsys_attribution_provenance.v2",
    }


def _run_stats(
    *,
    nsys_bin: Path,
    report_path: Path,
    report_name: str,
) -> str:
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    completed = subprocess.run(
        [
            os.fspath(nsys_bin),
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            "-",
            "--timeunit",
            "nsec",
            os.fspath(report_path),
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise ReductionError(
            f"nsys stats {report_name} failed with rc={completed.returncode}{suffix}"
        )
    return completed.stdout


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _strict_bool(raw: str) -> bool:
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise argparse.ArgumentTypeError("must be exactly 'true' or 'false'")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Reduce fixed32 Nsight GPU attribution without SWE task content")
    )
    parser.add_argument("report", type=Path, help="input .nsys-rep")
    parser.add_argument(
        "--output",
        type=Path,
        help="write canonical JSON here instead of stdout",
    )
    parser.add_argument(
        "--nsys-bin",
        type=Path,
        default=DEFAULT_NSYS_BIN,
        help=f"nsys executable (default: {DEFAULT_NSYS_BIN})",
    )
    parser.add_argument(
        "--top",
        type=_positive_int,
        default=20,
        help="number of kernels retained per ranking (default: 20)",
    )
    parser.add_argument("--subset", type=Path)
    parser.add_argument("--runtime-manifest-launch", type=Path)
    parser.add_argument("--runtime-manifest-end", type=Path)
    parser.add_argument("--external-manifest-launch", type=Path)
    parser.add_argument("--external-manifest-end", type=Path)
    parser.add_argument("--process-identity", type=Path)
    parser.add_argument("--container-identity", type=Path)
    parser.add_argument("--runtime-attestation", type=Path)
    parser.add_argument("--pretask-zero-traffic", type=Path)
    parser.add_argument("--proxy-ledger", type=Path)
    parser.add_argument("--engine-ledger", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--batch-size", type=_positive_int)
    parser.add_argument("--concurrency", type=_positive_int)
    parser.add_argument("--driver-rc", type=int)
    parser.add_argument("--nsys-delay-s", type=_positive_int)
    parser.add_argument("--nsys-duration-s", type=_positive_int)
    parser.add_argument("--nsys-flush-ms", type=_positive_int)
    parser.add_argument("--nsys-trace")
    parser.add_argument("--nsys-config-directives")
    parser.add_argument("--nsys-discard-environment", type=_strict_bool)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if not args.report.is_file():
            raise ReductionError("Nsight report does not exist")
        if not args.nsys_bin.is_file() or not os.access(args.nsys_bin, os.X_OK):
            raise ReductionError("nsys executable is missing or not executable")

        report_sha256, report_bytes = _sha256_file(args.report)
        stats_csv = {
            report_name: _run_stats(
                nsys_bin=args.nsys_bin,
                report_path=args.report,
                report_name=report_name,
            )
            for report_name in REPORT_NAMES
        }
        summary = _build_summary(
            report_sha256=report_sha256,
            report_bytes=report_bytes,
            stats_csv=stats_csv,
            top=args.top,
        )
        provenance_fields = (
            "subset",
            "runtime_manifest_launch",
            "runtime_manifest_end",
            "external_manifest_launch",
            "external_manifest_end",
            "process_identity",
            "container_identity",
            "runtime_attestation",
            "pretask_zero_traffic",
            "proxy_ledger",
            "engine_ledger",
            "mode",
            "batch_size",
            "concurrency",
            "driver_rc",
            "nsys_delay_s",
            "nsys_duration_s",
            "nsys_flush_ms",
            "nsys_trace",
            "nsys_config_directives",
            "nsys_discard_environment",
        )
        missing = [field for field in provenance_fields if getattr(args, field) is None]
        if missing:
            raise ReductionError(
                "publishable attribution requires provenance arguments: "
                + ", ".join("--" + field.replace("_", "-") for field in missing)
            )
        summary["provenance"] = _build_attribution_provenance(
            subset=args.subset,
            runtime_manifest_launch=args.runtime_manifest_launch,
            runtime_manifest_end=args.runtime_manifest_end,
            external_manifest_launch=args.external_manifest_launch,
            external_manifest_end=args.external_manifest_end,
            process_identity=args.process_identity,
            container_identity=args.container_identity,
            runtime_attestation=args.runtime_attestation,
            pretask_zero_traffic=args.pretask_zero_traffic,
            proxy_ledger=args.proxy_ledger,
            engine_ledger=args.engine_ledger,
            mode=args.mode,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            driver_rc=args.driver_rc,
            nsys_delay_s=args.nsys_delay_s,
            nsys_duration_s=args.nsys_duration_s,
            nsys_flush_ms=args.nsys_flush_ms,
            nsys_trace=args.nsys_trace,
            nsys_config_directives=args.nsys_config_directives,
            nsys_discard_environment=args.nsys_discard_environment,
            nsys_bin=args.nsys_bin,
        )
        summary["provenance_bound"] = True
        summary["curated_publishable"] = True
        rendered = (
            json.dumps(
                summary,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if args.output is None:
            sys.stdout.write(rendered)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
    except (KeyError, OSError, ReductionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
