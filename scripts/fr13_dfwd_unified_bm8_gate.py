#!/usr/bin/env python3
"""Validate the real-SWE B1 unified-attention BM8 byte gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
INSTANCE = re.compile(r"^[A-Za-z0-9._/-]+__[A-Za-z0-9._/-]+$")
IDENTITY_SCHEMA = "fr13.fixed32.dfwd_unified_bm8.identity.v1"
LIVE_SCHEMA = "fr13.fixed32.dfwd_unified_bm8_live_ab.v1"
EXPECTED_CANDIDATE = {
    "kernel": "kernel_unified_attention_2d",
    "stock_block_m": 16,
    "stock_block_q": 2,
    "candidate_block_m": 8,
    "candidate_block_q": 1,
    "required_calls": 4,
}
EXPECTED_FILES = {
    "patcher": "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "unified_attention": (
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "triton_unified_attention.py"
    ),
    "eagle_replay_hook": (
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
    ),
}


class GateError(RuntimeError):
    """The BM8 diagnostic did not produce an exact, attributable PASS."""


def _strict_object(raw: str, *, label: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate key {key!r}")
            value[key] = item
        return value

    def nonfinite(value: str) -> Any:
        raise ValueError(f"nonfinite value {value}")

    try:
        value = json.loads(
            raw,
            object_pairs_hook=object_pairs,
            parse_constant=nonfinite,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise GateError(f"{label} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} is not a JSON object")
    return value


def _load(path: Path, *, label: str, required_mode: int | None = None) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        raw = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise GateError(f"{label} is unavailable: {exc}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise GateError(f"{label} must be a single-link regular file")
    if required_mode is not None and stat.S_IMODE(metadata.st_mode) != required_mode:
        raise GateError(f"{label} mode is not {required_mode:04o}")
    if metadata.st_size != len(raw.encode("ascii")):
        raise GateError(f"{label} changed while reading")
    return _strict_object(raw, label=label)


def _validate_identity(
    identity: dict[str, Any], *, expected_source_commit: str
) -> None:
    if not HEX40.fullmatch(expected_source_commit):
        raise GateError("expected source commit is not a lowercase SHA-1")
    if (
        identity.get("schema") != IDENTITY_SCHEMA
        or identity.get("source_commit") != expected_source_commit
        or identity.get("production_enabled") is not False
        or identity.get("candidate") != EXPECTED_CANDIDATE
    ):
        raise GateError("candidate identity fields differ from the BM8 contract")
    files = identity.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise GateError("candidate identity file set differs from the BM8 contract")
    for label, expected_path in EXPECTED_FILES.items():
        row = files.get(label)
        if (
            not isinstance(row, dict)
            or row.get("path") != expected_path
            or not isinstance(row.get("sha256"), str)
            or HEX64.fullmatch(row["sha256"]) is None
        ):
            raise GateError(f"candidate identity is invalid for {label}")


def verify(
    *,
    live_result: Path,
    identity_path: Path,
    expected_source_commit: str,
    expected_instance_id: str,
) -> dict[str, Any]:
    if INSTANCE.fullmatch(expected_instance_id) is None:
        raise GateError("expected instance ID is not canonical")
    identity = _load(
        identity_path,
        label="candidate identity",
        required_mode=0o400,
    )
    _validate_identity(identity, expected_source_commit=expected_source_commit)
    live = _load(live_result, label="live result")
    if live.get("status") != "PASS":
        raise GateError(
            "BM8 live result is not PASS: "
            + str(live.get("error", live.get("status")))
        )
    if (
        live.get("schema") != LIVE_SCHEMA
        or live.get("suite") != "SWE-Verified"
        or live.get("instance_id") != expected_instance_id
        or live.get("task_marker") != f"swe_verified:{expected_instance_id}"
        or live.get("concurrency") != 1
        or live.get("batch_size") != 1
        or live.get("candidate_identity") != identity
        or live.get("candidate_dispatch")
        != "launcher-private BM8 exact B1 selector"
        or live.get("candidate_dispatches") != 4
        or live.get("served_return")
        != "stock captured drafter graph unchanged"
        or live.get("performance_measurement") is not False
    ):
        raise GateError("BM8 live result provenance or dispatch fields drifted")
    expected_geometry = {
        "query_shape": [1, 24, 256],
        "kv_heads": 4,
        "stock_block_m": 16,
        "stock_block_q": 2,
        "candidate_block_m": 8,
        "candidate_block_q": 1,
        "valid_query_heads_per_kv": 6,
    }
    if live.get("geometry") != expected_geometry:
        raise GateError("BM8 live result geometry drifted")
    calls = live.get("calls")
    if not isinstance(calls, list) or len(calls) != 4:
        raise GateError("BM8 live result does not contain four calls")
    for index, row in enumerate(calls):
        if (
            not isinstance(row, dict)
            or row.get("call_index") != index
            or type(row.get("seq_len")) is not int
            or row["seq_len"] <= 0
            or row.get("bytes") != 1 * 24 * 256 * 2
            or row.get("raw_byte_mismatches") != 0
            or not isinstance(row.get("stock_sha256"), str)
            or HEX64.fullmatch(row["stock_sha256"]) is None
            or row.get("candidate_sha256") != row.get("stock_sha256")
        ):
            raise GateError(f"BM8 live call {index} is not an exact byte PASS")
    return {
        "schema": "fr13.fixed32.dfwd_unified_bm8.validation.v1",
        "status": "PASS",
        "source_commit": expected_source_commit,
        "instance_id": expected_instance_id,
        "calls": len(calls),
        "candidate_dispatches": live["candidate_dispatches"],
        "raw_byte_mismatches": sum(row["raw_byte_mismatches"] for row in calls),
        "identity_sha256": hashlib.sha256(identity_path.read_bytes()).hexdigest(),
        "live_result_sha256": hashlib.sha256(live_result.read_bytes()).hexdigest(),
        "performance_measurement": False,
        "production_enabled": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("verify")
    command.add_argument("--live-result", type=Path, required=True)
    command.add_argument("--identity", type=Path, required=True)
    command.add_argument("--expected-source-commit", required=True)
    command.add_argument("--expected-instance-id", required=True)
    args = parser.parse_args()
    try:
        result = verify(
            live_result=args.live_result,
            identity_path=args.identity,
            expected_source_commit=args.expected_source_commit,
            expected_instance_id=args.expected_instance_id,
        )
    except GateError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
