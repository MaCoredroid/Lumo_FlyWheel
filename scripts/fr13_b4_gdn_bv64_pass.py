#!/usr/bin/env python3
"""Validate and install an exact4 B4 BV64 graph-byte PASS artifact."""

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
EXPECTED_CANDIDATE = "fixed32_batch_gdn_bv_v2"
EXPECTED_MODE = "tail6_fixed32"
EXPECTED_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
EXPECTED_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
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
EXPECTED_KEYS = {
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
    "compared_byte_surfaces",
    "raw_byte_equal",
    "state_restored",
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
    "candidate_bv",
    "b4_layer_passes",
    "observed_pass_layers_by_batch",
    "engine_ledger_chain_head_sha256",
    "graph_live_pass_sha256",
    "kernel_source_sha256",
    "raw_byte_equal",
    "reference_always_served",
    "production_default_enabled",
}
HEX = frozenset("0123456789abcdef")
READ_CHUNK = 1024 * 1024
MAX_PASS_BYTES = 1024 * 1024


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


def _load_json_line(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, max_bytes=MAX_PASS_BYTES)
    if not raw.endswith(b"\n") or raw.count(b"\n") != 1:
        raise ValueError(f"{label} must be one ASCII JSON line with a newline")
    try:
        text = raw.decode("ascii")
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not canonical ASCII JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be an object")
    return payload, raw


def validate_live_result(
    payload: dict[str, Any],
    *,
    kernel_source_sha256: str,
) -> dict[str, Any]:
    kernel_source_sha256 = _require_sha256(
        kernel_source_sha256, "kernel source"
    )
    if set(payload) != EXPECTED_KEYS:
        missing = sorted(EXPECTED_KEYS - set(payload))
        extra = sorted(set(payload) - EXPECTED_KEYS)
        raise ValueError(
            f"graph PASS key set drifted: missing={missing!r} extra={extra!r}"
        )
    task_marker = payload.get("task_marker")
    task_prefix = "swe_verified:"
    task_id = (
        task_marker[len(task_prefix) :]
        if isinstance(task_marker, str) and task_marker.startswith(task_prefix)
        else ""
    )
    if not task_id or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        for character in task_id
    ):
        raise ValueError("graph PASS is not bound to a real SWE-Verified task")
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
        "candidate_bv": 64,
        "reference_physical_launches_per_layer": 8,
        "candidate_physical_launches_per_layer": 2,
        "compared_byte_surfaces": list(EXPECTED_SURFACES),
        "raw_byte_equal": True,
        "state_restored": True,
        "gate_mode": "post_replay_shadow",
        "capture_records": 48,
        "real_task_authenticated": True,
        "graph_baseline_byte_equal": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"graph PASS field drifted: {key}")
    graph_id = payload.get("graph_id")
    if type(graph_id) is not int or graph_id <= 0:
        raise ValueError("graph PASS graph_id is invalid")
    graph_signature = _require_sha256(
        payload.get("graph_signature"), "graph signature"
    )
    layer_keys = payload.get("layer_keys")
    if not isinstance(layer_keys, list) or len(layer_keys) != 48:
        raise ValueError("graph PASS must cover exactly 48 layers")
    try:
        numeric_keys = [int(key, 16) for key in layer_keys]
    except (TypeError, ValueError) as error:
        raise ValueError("graph PASS layer keys are not hexadecimal") from error
    if (
        len(set(numeric_keys)) != 48
        or numeric_keys != sorted(numeric_keys)
        or layer_keys != [f"0x{key:x}" for key in numeric_keys]
    ):
        raise ValueError("graph PASS layer keys are not distinct canonical hex")
    return {
        "schema": LIVE_SCHEMA,
        "task_marker": task_marker,
        "graph_id": graph_id,
        "graph_signature": graph_signature,
        "candidate_bv": 64,
        "layer_count": 48,
        "source_sha256": kernel_source_sha256,
    }


def validate_gate_verdict(
    payload: dict[str, Any],
    *,
    live_summary: dict[str, Any],
    live_result_sha256: str,
    kernel_source_sha256: str,
) -> dict[str, Any]:
    if set(payload) != EXPECTED_VERDICT_KEYS:
        missing = sorted(EXPECTED_VERDICT_KEYS - set(payload))
        extra = sorted(set(payload) - EXPECTED_VERDICT_KEYS)
        raise ValueError(
            f"graph gate verdict key set drifted: missing={missing!r} extra={extra!r}"
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
        "candidate_bv": 64,
        "b4_layer_passes": 48,
        "observed_pass_layers_by_batch": {"2": 0, "3": 0, "4": 48},
        "graph_live_pass_sha256": live_result_sha256,
        "kernel_source_sha256": kernel_source_sha256,
        "raw_byte_equal": True,
        "reference_always_served": True,
        "production_default_enabled": False,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"graph gate verdict field drifted: {key}")
    ledger_sha256 = _require_sha256(
        payload.get("engine_ledger_chain_head_sha256"),
        "engine ledger chain head",
    )
    return {
        "schema": GATE_VERDICT_SCHEMA,
        "subset_sha256": EXPECTED_SUBSET_SHA256,
        "task_ids": list(EXPECTED_TASK_IDS),
        "engine_ledger_chain_head_sha256": ledger_sha256,
    }


def validate_file(
    *,
    live_result: Path,
    expected_live_sha256: str,
    gate_verdict: Path,
    expected_gate_verdict_sha256: str,
    kernel_source: Path,
) -> tuple[dict[str, Any], bytes]:
    expected_live_sha256 = _require_sha256(
        expected_live_sha256, "graph PASS artifact"
    )
    expected_gate_verdict_sha256 = _require_sha256(
        expected_gate_verdict_sha256, "graph gate verdict"
    )
    payload, raw = _load_json_line(live_result, "graph PASS")
    actual_live_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_live_sha256 != expected_live_sha256:
        raise ValueError("graph PASS raw SHA-256 mismatch")
    kernel_raw = _read_regular(kernel_source)
    kernel_source_sha256 = hashlib.sha256(kernel_raw).hexdigest()
    summary = validate_live_result(
        payload,
        kernel_source_sha256=kernel_source_sha256,
    )
    verdict_payload, verdict_raw = _load_json_line(
        gate_verdict, "graph gate verdict"
    )
    actual_verdict_sha256 = hashlib.sha256(verdict_raw).hexdigest()
    if actual_verdict_sha256 != expected_gate_verdict_sha256:
        raise ValueError("graph gate verdict raw SHA-256 mismatch")
    verdict_summary = validate_gate_verdict(
        verdict_payload,
        live_summary=summary,
        live_result_sha256=actual_live_sha256,
        kernel_source_sha256=kernel_source_sha256,
    )
    summary["live_result_sha256"] = actual_live_sha256
    summary["gate_verdict_sha256"] = actual_verdict_sha256
    summary["exact4_subset_sha256"] = verdict_summary["subset_sha256"]
    return summary, raw


def _install_bytes(raw: bytes, out: Path) -> None:
    parent = out.parent
    parent_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    parent_flags |= getattr(os, "O_DIRECTORY", 0)
    parent_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        parent_fd = os.open(parent, parent_flags)
    except OSError as error:
        raise ValueError(f"cannot securely open output directory {parent}: {error}") from error
    temporary = f".{out.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}"
    descriptor = None
    published = False
    try:
        try:
            os.stat(out.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"refusing to replace graph PASS: {out}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary, flags, 0o600, dir_fd=parent_fd)
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("short write while installing graph PASS")
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
    out: Path,
) -> dict[str, Any]:
    summary, raw = validate_file(
        live_result=live_result,
        expected_live_sha256=expected_live_sha256,
        gate_verdict=gate_verdict,
        expected_gate_verdict_sha256=expected_gate_verdict_sha256,
        kernel_source=kernel_source,
    )
    _install_bytes(raw, out)
    return {**summary, "installed_path": str(out)}


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
        if command == "install":
            child.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "install":
            summary = install_pass(
                live_result=args.live_result,
                expected_live_sha256=args.expected_live_sha256,
                gate_verdict=args.gate_verdict,
                expected_gate_verdict_sha256=(
                    args.expected_gate_verdict_sha256
                ),
                kernel_source=args.kernel_source,
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
            )
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
