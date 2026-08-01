#!/usr/bin/env python3
"""Validate and install an exact4 B4 batched-BV8 production credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.batch_gdn.graph_live_pass.v1"
GATE_VERDICT_SCHEMA = "fr13.fixed32.batch_gdn.b4_diagnostic.v1"
SIDECAR_SCHEMA = "fr13.fixed32.batch_gdn.bv8.production_sidecar.v1"
ENGAGEMENT_SCHEMA = "fr13.fixed32.batch_gdn.bv8.production_engagement.v1"
RUNTIME_MANIFEST_SCHEMA = "fr13-runtime-manifest-v1"
RUNTIME_MANIFEST_FORMAT = "utf8-json-sort-keys-compact-v1"
EXPECTED_CANDIDATE = "fixed32_batch_gdn_bv8_v1"
EXPECTED_MODE = "tail6_fixed32"
EXPECTED_PRODUCTION_MODE = "hydra27_fixed32"
EXPECTED_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
REFERENCE_STRUCTURE = "per_request_tree_gdn_path"
CANDIDATE_STRUCTURE = "fixed32_batch_tree_gdn_path"
GATE_RUNNER_PATH = "scripts/fr13_run_b4_gdn_wide_live_gate.sh"
TIMING_RUNNER_PATH = "scripts/fr13_run_b4_gdn_bv8_timing.sh"
EXPECTED_SURFACES = (
    "out",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "state_export_compact",
    "state_export_untouched_tail",
    "flags",
    "invocation_counter",
)
EXPECTED_LIVE_KEYS = {
    "schema",
    "status",
    "task_marker",
    "batch",
    "layer_count",
    "layer_keys",
    "reference_always_served",
    "candidate",
    "source_sha256",
    "mode",
    "physical_rows_per_request",
    "reference_bv",
    "candidate_bv",
    "reference_physical_launches_per_layer",
    "candidate_physical_launches_per_layer",
    "count_invocation",
    "ring_export",
    "flags_inkernel",
    "scan_align",
    "npad_invariant",
    "compared_byte_surfaces",
    "raw_byte_equal",
    "state_restored",
    "reference_kernel_structure",
    "candidate_kernel_structure",
    "production_eligible",
    "gate_mode",
    "graph_id",
    "graph_signature",
    "capture_records",
    "real_task_authenticated",
    "graph_baseline_byte_equal",
}
EXPECTED_VERDICT_KEYS = {
    "schema",
    "status",
    "run_classification",
    "timing_eligible",
    "floor_acceptance_eligible",
    "subset_sha256",
    "task_ids",
    "task_marker",
    "gate_mode",
    "graph_id",
    "graph_signature",
    "candidate",
    "reference_bv",
    "candidate_bv",
    "reference_kernel_structure",
    "candidate_kernel_structure",
    "reference_physical_launches_per_layer",
    "candidate_physical_launches_per_layer",
    "count_invocation",
    "ring_export",
    "flags_inkernel",
    "scan_align",
    "npad_invariant",
    "tree_gdn_geom_override",
    "enforce_eager",
    "cudagraph_mode",
    "production_eligible",
    "b4_layer_passes",
    "observed_pass_layers_by_batch",
    "engine_ledger_chain_head_sha256",
    "graph_live_pass_sha256",
    "kernel_source_sha256",
    "runtime_manifest_sha256",
    "gate_runner_sha256",
    "raw_byte_equal",
    "reference_always_served",
    "production_default_enabled",
}
EXPECTED_ENGAGEMENT_KEYS = {
    "schema",
    "status",
    "mode",
    "runtime_mode",
    "selector",
    "batch_size",
    "candidate",
    "reference_kernel_structure",
    "candidate_kernel_structure",
    "reference_bv",
    "candidate_bv",
    "reference_physical_launches_per_layer",
    "candidate_physical_launches_per_layer",
    "count_invocation",
    "ring_export",
    "flags_inkernel",
    "scan_align",
    "npad_invariant",
    "physical_rows_per_request",
    "layer_count",
    "layer_keys",
    "batched_route_capture_layers_by_batch",
    "qualified_batch_sizes",
    "lower_batch_route",
    "physical_launches_per_layer_by_batch",
    "all_b_le_4_launch_invariant",
    "graph_id",
    "graph_signature",
    "graph_pass_sha256",
    "gate_verdict_sha256",
    "production_sidecar_sha256",
    "kernel_source_sha256",
    "runtime_manifest_sha256",
    "gate_runner_sha256",
    "task_marker",
    "observed_full_graph_replays_at_least",
    "fallback",
    "production_default_enabled",
}
HEX = frozenset("0123456789abcdef")
READ_CHUNK = 1024 * 1024
MAX_ARTIFACT_BYTES = 1024 * 1024


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read_regular(path: Path, *, max_bytes: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"cannot securely open {path}: {error}") from error
    chunks: list[bytes] = []
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, READ_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise ValueError(f"file exceeds {max_bytes} bytes: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _identity(before) != _identity(after) or total != before.st_size:
        raise ValueError(f"file changed while being read: {path}")
    try:
        current = path.lstat()
    except OSError as error:
        raise ValueError(f"file disappeared after being read: {path}") from error
    if stat.S_ISLNK(current.st_mode) or _identity(before) != _identity(current):
        raise ValueError(f"file identity changed while being read: {path}")
    return b"".join(chunks)


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_raw(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _load_json_line(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, max_bytes=MAX_ARTIFACT_BYTES)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{label} must be one canonical ASCII JSON line")
    try:
        payload = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    if raw != _canonical_raw(payload):
        raise ValueError(f"{label} is not canonical sorted compact JSON")
    return payload, raw


def _load_json_document(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, max_bytes=MAX_ARTIFACT_BYTES)
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not UTF-8 JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, raw


def _canonical_digest(payload: object) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_runtime_closure(
    runtime_manifest: Path, gate_runner: Path
) -> dict[str, str]:
    manifest, _raw = _load_json_document(runtime_manifest, "runtime manifest")
    expected_manifest_keys = {
        "canonical_format",
        "closures",
        "overall_canonical_sha256",
        "profile",
        "required_absence",
        "schema",
        "sequence",
        "summary",
    }
    if set(manifest) != expected_manifest_keys:
        raise ValueError("runtime manifest key set drifted")
    recorded_digest = _require_sha256(
        manifest.get("overall_canonical_sha256"), "runtime manifest"
    )
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key != "overall_canonical_sha256"
    }
    if _canonical_digest(unsigned) != recorded_digest:
        raise ValueError("runtime manifest canonical digest mismatch")
    if (
        manifest.get("schema") != RUNTIME_MANIFEST_SCHEMA
        or manifest.get("canonical_format") != RUNTIME_MANIFEST_FORMAT
        or manifest.get("profile") != "fixed32"
        or manifest.get("sequence")
        != "scripts/fr13_fixed32_floor_timers_seq.sh"
    ):
        raise ValueError("runtime manifest fixed32 identity drifted")
    closures = manifest.get("closures")
    host_records = (
        closures.get("host_script_source")
        if isinstance(closures, dict)
        else None
    )
    if not isinstance(host_records, list):
        raise ValueError("runtime manifest host-script closure is missing")
    records_by_path: dict[str, dict[str, Any]] = {}
    for record in host_records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "sha256", "size"}
            or not isinstance(record.get("path"), str)
            or record["path"] in records_by_path
        ):
            raise ValueError("runtime manifest host-script record drifted")
        records_by_path[record["path"]] = record
    if GATE_RUNNER_PATH not in records_by_path:
        raise ValueError("runtime manifest does not close over the gate runner")
    if TIMING_RUNNER_PATH not in records_by_path:
        raise ValueError("runtime manifest does not close over the timing runner")
    gate_runner_raw = _read_regular(gate_runner)
    gate_runner_sha256 = hashlib.sha256(gate_runner_raw).hexdigest()
    gate_record = records_by_path[GATE_RUNNER_PATH]
    if (
        _require_sha256(gate_record.get("sha256"), "manifest gate runner")
        != gate_runner_sha256
        or gate_record.get("size") != len(gate_runner_raw)
    ):
        raise ValueError("gate runner differs from the runtime manifest closure")
    _require_sha256(
        records_by_path[TIMING_RUNNER_PATH].get("sha256"),
        "manifest timing runner",
    )
    return {
        "runtime_manifest_sha256": recorded_digest,
        "gate_runner_sha256": gate_runner_sha256,
    }


def _validate_layer_keys(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or len(value) != 48:
        raise ValueError(f"{label} must cover exactly 48 layers")
    try:
        numeric = [int(key, 16) for key in value]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} layer keys are not hexadecimal") from error
    if (
        any(key <= 0 for key in numeric)
        or len(set(numeric)) != 48
        or numeric != sorted(numeric)
        or value != [f"0x{key:x}" for key in numeric]
    ):
        raise ValueError(f"{label} layer keys are not distinct canonical hex")
    return value


def validate_live_result(
    payload: dict[str, Any], *, kernel_source_sha256: str
) -> dict[str, Any]:
    kernel_source_sha256 = _require_sha256(kernel_source_sha256, "kernel source")
    if set(payload) != EXPECTED_LIVE_KEYS:
        raise ValueError(
            "graph PASS key set drifted: "
            f"missing={sorted(EXPECTED_LIVE_KEYS - set(payload))!r} "
            f"extra={sorted(set(payload) - EXPECTED_LIVE_KEYS)!r}"
        )
    task_marker = payload.get("task_marker")
    valid_markers = {f"swe_verified:{task_id}" for task_id in EXPECTED_TASK_IDS}
    expected = {
        "schema": LIVE_SCHEMA,
        "status": "pass",
        "batch": 4,
        "layer_count": 48,
        "reference_always_served": True,
        "candidate": EXPECTED_CANDIDATE,
        "source_sha256": kernel_source_sha256,
        "mode": EXPECTED_MODE,
        "physical_rows_per_request": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
        "compared_byte_surfaces": list(EXPECTED_SURFACES),
        "raw_byte_equal": True,
        "state_restored": True,
        "reference_kernel_structure": REFERENCE_STRUCTURE,
        "candidate_kernel_structure": CANDIDATE_STRUCTURE,
        "production_eligible": True,
        "gate_mode": "post_replay_shadow",
        "capture_records": 48,
        "real_task_authenticated": True,
        "graph_baseline_byte_equal": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"graph PASS field drifted: {key}")
    if task_marker not in valid_markers:
        raise ValueError("graph PASS is not bound to the canonical exact4 tasks")
    graph_id = payload.get("graph_id")
    if type(graph_id) is not int or graph_id <= 0:
        raise ValueError("graph PASS graph_id is invalid")
    graph_signature = _require_sha256(
        payload.get("graph_signature"), "graph signature"
    )
    _validate_layer_keys(payload.get("layer_keys"), "graph PASS")
    return {
        "task_marker": task_marker,
        "graph_id": graph_id,
        "graph_signature": graph_signature,
        "source_sha256": kernel_source_sha256,
    }


def validate_gate_verdict(
    payload: dict[str, Any],
    *,
    live_summary: dict[str, Any],
    live_result_sha256: str,
    kernel_source_sha256: str,
    runtime_manifest_sha256: str,
    gate_runner_sha256: str,
) -> dict[str, Any]:
    if set(payload) != EXPECTED_VERDICT_KEYS:
        raise ValueError(
            "graph gate verdict key set drifted: "
            f"missing={sorted(EXPECTED_VERDICT_KEYS - set(payload))!r} "
            f"extra={sorted(set(payload) - EXPECTED_VERDICT_KEYS)!r}"
        )
    expected = {
        "schema": GATE_VERDICT_SCHEMA,
        "status": "pass",
        "run_classification": "exact4_b4_graph_byte_diagnostic",
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_marker": live_summary["task_marker"],
        "gate_mode": "post_replay_shadow",
        "graph_id": live_summary["graph_id"],
        "graph_signature": live_summary["graph_signature"],
        "candidate": EXPECTED_CANDIDATE,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_kernel_structure": REFERENCE_STRUCTURE,
        "candidate_kernel_structure": CANDIDATE_STRUCTURE,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
        "tree_gdn_geom_override": "BV=8",
        "enforce_eager": 0,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "production_eligible": True,
        "b4_layer_passes": 48,
        "observed_pass_layers_by_batch": {"2": 0, "3": 0, "4": 48},
        "graph_live_pass_sha256": live_result_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "gate_runner_sha256": gate_runner_sha256,
        "raw_byte_equal": True,
        "reference_always_served": True,
        "production_default_enabled": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"graph gate verdict field drifted: {key}")
    _require_sha256(
        payload.get("engine_ledger_chain_head_sha256"),
        "engine ledger chain head",
    )
    return payload


def validate_file(
    *,
    live_result: Path,
    expected_live_sha256: str,
    gate_verdict: Path,
    expected_gate_verdict_sha256: str,
    kernel_source: Path,
    runtime_manifest: Path,
    gate_runner: Path,
) -> tuple[dict[str, Any], bytes]:
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "graph PASS artifact"
    )
    expected_gate_verdict_sha256 = _require_sha256(
        expected_gate_verdict_sha256, "graph gate verdict"
    )
    live_payload, live_raw = _load_json_line(live_result, "graph PASS")
    if hashlib.sha256(live_raw).hexdigest() != expected_live_sha256:
        raise ValueError("graph PASS raw SHA-256 mismatch")
    kernel_source_sha256 = hashlib.sha256(_read_regular(kernel_source)).hexdigest()
    live_summary = validate_live_result(
        live_payload, kernel_source_sha256=kernel_source_sha256
    )
    closure = validate_runtime_closure(runtime_manifest, gate_runner)
    verdict_payload, verdict_raw = _load_json_line(
        gate_verdict, "graph gate verdict"
    )
    if hashlib.sha256(verdict_raw).hexdigest() != expected_gate_verdict_sha256:
        raise ValueError("graph gate verdict raw SHA-256 mismatch")
    validate_gate_verdict(
        verdict_payload,
        live_summary=live_summary,
        live_result_sha256=expected_live_sha256,
        kernel_source_sha256=kernel_source_sha256,
        runtime_manifest_sha256=closure["runtime_manifest_sha256"],
        gate_runner_sha256=closure["gate_runner_sha256"],
    )
    credential = {
        "schema": SIDECAR_SCHEMA,
        "status": "qualified",
        "candidate": EXPECTED_CANDIDATE,
        "batch": 4,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "task_ids": list(EXPECTED_TASK_IDS),
        "task_marker": live_summary["task_marker"],
        "kernel_source_sha256": kernel_source_sha256,
        "runtime_manifest_sha256": closure["runtime_manifest_sha256"],
        "gate_runner_sha256": closure["gate_runner_sha256"],
        "live_result_sha256": expected_live_sha256,
        "gate_verdict_sha256": expected_gate_verdict_sha256,
        "reference_kernel_structure": REFERENCE_STRUCTURE,
        "candidate_kernel_structure": CANDIDATE_STRUCTURE,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "production_default_enabled": False,
        "live_result": live_payload,
        "gate_verdict": verdict_payload,
    }
    summary = {
        "schema": SIDECAR_SCHEMA,
        "status": "qualified",
        "candidate": EXPECTED_CANDIDATE,
        "task_marker": live_summary["task_marker"],
        "graph_id": live_summary["graph_id"],
        "graph_signature": live_summary["graph_signature"],
        "kernel_source_sha256": kernel_source_sha256,
        "runtime_manifest_sha256": closure["runtime_manifest_sha256"],
        "gate_runner_sha256": closure["gate_runner_sha256"],
        "live_result_sha256": expected_live_sha256,
        "gate_verdict_sha256": expected_gate_verdict_sha256,
    }
    return summary, _canonical_raw(credential)


def _install_bytes(raw: bytes, out: Path) -> None:
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    parent_flags |= getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(out.parent, parent_flags)
    except OSError as error:
        raise ValueError(
            f"cannot securely open output directory {out.parent}: {error}"
        ) from error
    temporary = f".{out.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    descriptor = None
    published = False
    try:
        try:
            os.stat(out.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"refusing to replace production sidecar: {out}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write while installing production sidecar")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(
            temporary,
            out.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        if published:
            try:
                os.unlink(out.name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        os.close(parent_fd)


def install_pass(
    *,
    live_result: Path,
    expected_live_sha256: str,
    gate_verdict: Path,
    expected_gate_verdict_sha256: str,
    kernel_source: Path,
    runtime_manifest: Path,
    gate_runner: Path,
    out: Path,
) -> dict[str, Any]:
    summary, credential_raw = validate_file(
        live_result=live_result,
        expected_live_sha256=expected_live_sha256,
        gate_verdict=gate_verdict,
        expected_gate_verdict_sha256=expected_gate_verdict_sha256,
        kernel_source=kernel_source,
        runtime_manifest=runtime_manifest,
        gate_runner=gate_runner,
    )
    _install_bytes(credential_raw, out)
    return {
        **summary,
        "production_sidecar_sha256": hashlib.sha256(credential_raw).hexdigest(),
        "installed_path": str(out),
    }


def validate_engagement_file(
    *,
    engagement: Path,
    expected_live_sha256: str,
    expected_gate_verdict_sha256: str,
    expected_production_sidecar_sha256: str,
    expected_runtime_manifest_sha256: str,
    expected_gate_runner_sha256: str,
    kernel_source: Path,
) -> tuple[dict[str, Any], bytes]:
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "graph PASS artifact"
    )
    expected_gate_verdict_sha256 = _require_sha256(
        expected_gate_verdict_sha256, "graph gate verdict"
    )
    expected_production_sidecar_sha256 = _require_sha256(
        expected_production_sidecar_sha256, "production sidecar"
    )
    expected_runtime_manifest_sha256 = _require_sha256(
        expected_runtime_manifest_sha256, "runtime manifest"
    )
    expected_gate_runner_sha256 = _require_sha256(
        expected_gate_runner_sha256, "gate runner"
    )
    payload, raw = _load_json_line(engagement, "batched BV8 engagement")
    if set(payload) != EXPECTED_ENGAGEMENT_KEYS:
        raise ValueError(
            "batched BV8 engagement key set drifted: "
            f"missing={sorted(EXPECTED_ENGAGEMENT_KEYS - set(payload))!r} "
            f"extra={sorted(set(payload) - EXPECTED_ENGAGEMENT_KEYS)!r}"
        )
    kernel_source_sha256 = hashlib.sha256(_read_regular(kernel_source)).hexdigest()
    expected = {
        "schema": ENGAGEMENT_SCHEMA,
        "status": "ENGAGED",
        "mode": EXPECTED_PRODUCTION_MODE,
        "runtime_mode": "FULL",
        "selector": "production",
        "batch_size": 4,
        "candidate": EXPECTED_CANDIDATE,
        "reference_kernel_structure": REFERENCE_STRUCTURE,
        "candidate_kernel_structure": CANDIDATE_STRUCTURE,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "count_invocation": True,
        "ring_export": True,
        "flags_inkernel": True,
        "scan_align": False,
        "npad_invariant": False,
        "physical_rows_per_request": 32,
        "layer_count": 48,
        "batched_route_capture_layers_by_batch": {
            "1": 0,
            "2": 48,
            "3": 48,
            "4": 48,
        },
        "qualified_batch_sizes": [4],
        "lower_batch_route": "b1_legacy_b2_b3_fixed32_batched_bv8",
        "physical_launches_per_layer_by_batch": {
            "1": 2,
            "2": 2,
            "3": 2,
            "4": 2,
        },
        "all_b_le_4_launch_invariant": True,
        "graph_pass_sha256": expected_live_sha256,
        "gate_verdict_sha256": expected_gate_verdict_sha256,
        "production_sidecar_sha256": expected_production_sidecar_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "runtime_manifest_sha256": expected_runtime_manifest_sha256,
        "gate_runner_sha256": expected_gate_runner_sha256,
        "observed_full_graph_replays_at_least": 1,
        "fallback": 0,
        "production_default_enabled": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"batched BV8 engagement field drifted: {key}")
    if payload.get("task_marker") not in {
        f"swe_verified:{task_id}" for task_id in EXPECTED_TASK_IDS
    }:
        raise ValueError("batched BV8 engagement task marker drifted")
    graph_id = payload.get("graph_id")
    if type(graph_id) is not int or graph_id <= 0:
        raise ValueError("batched BV8 engagement graph_id is invalid")
    graph_signature = _require_sha256(
        payload.get("graph_signature"), "engagement graph signature"
    )
    _validate_layer_keys(payload.get("layer_keys"), "batched BV8 engagement")
    return {
        "schema": ENGAGEMENT_SCHEMA,
        "status": "ENGAGED",
        "graph_id": graph_id,
        "graph_signature": graph_signature,
        "graph_pass_sha256": expected_live_sha256,
        "gate_verdict_sha256": expected_gate_verdict_sha256,
        "production_sidecar_sha256": expected_production_sidecar_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "runtime_manifest_sha256": expected_runtime_manifest_sha256,
        "gate_runner_sha256": expected_gate_runner_sha256,
        "b4_replays_at_least": 1,
        "lower_batch_batched_capture_layers": 96,
    }, raw


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "install"):
        child = subparsers.add_parser(command)
        child.add_argument("--live-result", required=True, type=Path)
        child.add_argument("--expected-live-sha256", required=True)
        child.add_argument("--gate-verdict", required=True, type=Path)
        child.add_argument("--expected-gate-verdict-sha256", required=True)
        child.add_argument("--kernel-source", required=True, type=Path)
        child.add_argument("--runtime-manifest", required=True, type=Path)
        child.add_argument("--gate-runner", required=True, type=Path)
        if command == "install":
            child.add_argument("--out", required=True, type=Path)
    engagement = subparsers.add_parser("engagement")
    engagement.add_argument("--engagement", required=True, type=Path)
    engagement.add_argument("--expected-live-sha256", required=True)
    engagement.add_argument("--expected-gate-verdict-sha256", required=True)
    engagement.add_argument(
        "--expected-production-sidecar-sha256", required=True
    )
    engagement.add_argument("--expected-runtime-manifest-sha256", required=True)
    engagement.add_argument("--expected-gate-runner-sha256", required=True)
    engagement.add_argument("--kernel-source", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "engagement":
            summary, _raw = validate_engagement_file(
                engagement=args.engagement,
                expected_live_sha256=args.expected_live_sha256,
                expected_gate_verdict_sha256=(
                    args.expected_gate_verdict_sha256
                ),
                expected_production_sidecar_sha256=(
                    args.expected_production_sidecar_sha256
                ),
                expected_runtime_manifest_sha256=(
                    args.expected_runtime_manifest_sha256
                ),
                expected_gate_runner_sha256=args.expected_gate_runner_sha256,
                kernel_source=args.kernel_source,
            )
        elif args.command == "install":
            summary = install_pass(
                live_result=args.live_result,
                expected_live_sha256=args.expected_live_sha256,
                gate_verdict=args.gate_verdict,
                expected_gate_verdict_sha256=(
                    args.expected_gate_verdict_sha256
                ),
                kernel_source=args.kernel_source,
                runtime_manifest=args.runtime_manifest,
                gate_runner=args.gate_runner,
                out=args.out,
            )
        else:
            summary, _raw = validate_file(
                live_result=args.live_result,
                expected_live_sha256=args.expected_live_sha256,
                gate_verdict=args.gate_verdict,
                expected_gate_verdict_sha256=(
                    args.expected_gate_verdict_sha256
                ),
                kernel_source=args.kernel_source,
                runtime_manifest=args.runtime_manifest,
                gate_runner=args.gate_runner,
            )
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
