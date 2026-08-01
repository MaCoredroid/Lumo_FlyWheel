#!/usr/bin/env python3
"""Validate a source-bound fixed32 GDN coefficient-staging live PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "fr13.fixed32.gdn_level0_coeff.live_pass.v1"
CANDIDATE = "fixed32_gdn_level0_coeff_v1"
# Per layer: output 393,216 + 31 FP32 export rows 97,517,568 + raw
# K/V/A/B rings 530,432 + flags/counter 12 = 98,441,228 bytes; 48 layers.
EXPECTED_COMPARED_BYTES = 4_725_178_944
SURFACES = [
    "output",
    "export_non_scratch_rows",
    "ring_k",
    "ring_v",
    "ring_a",
    "ring_b",
    "flags",
    "counter",
]


class PassError(ValueError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise PassError(f"live PASS must be a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PassError(f"cannot read live PASS {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PassError("live PASS must be a JSON object")
    return payload


def validate(
    payload: dict[str, object],
    *,
    source_sha256: str,
    expected_task_id: str,
    expected_mode: str,
) -> dict[str, object]:
    expected = {
        "schema": SCHEMA,
        "status": "pass",
        "candidate": CANDIDATE,
        "source_sha256": source_sha256,
        "task_marker": f"swe_verified:{expected_task_id}",
        "mode": expected_mode,
        "batch_size": 1,
        "covered_batches": [1],
        "records": 48,
        "physical_rows": 32,
        "path_lengths": [5, 7],
        "launches_per_layer": 2,
        "scratch_row_start": 31,
        "scratch_rows": 1,
        "count_invocation": False,
        "non_scratch_export_rows_compared": 31,
        "surfaces": SURFACES,
        "raw_byte_equal": True,
        "scratch_contained": True,
        "reference_served": True,
        "state_restored": True,
    }
    drift = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        raise PassError(f"live PASS contract drift: {drift!r}")
    if expected_mode not in ("tail6_fixed32", "hydra27_fixed32"):
        raise PassError("expected mode is not an exact fixed32 mode")
    compared_bytes = payload.get("compared_bytes")
    if compared_bytes != EXPECTED_COMPARED_BYTES:
        raise PassError(
            "live PASS compared-byte closure drift: "
            f"{compared_bytes!r} != {EXPECTED_COMPARED_BYTES}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-result", type=Path, required=True)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--expected-task-id", required=True)
    parser.add_argument(
        "--expected-mode",
        choices=("tail6_fixed32", "hydra27_fixed32"),
        required=True,
    )
    parser.add_argument("--expected-live-sha256")
    args = parser.parse_args()
    if args.expected_live_sha256 and _sha256(args.live_result) != args.expected_live_sha256:
        raise PassError("live PASS SHA256 differs from the pinned identity")
    payload = validate(
        load(args.live_result),
        source_sha256=_sha256(args.kernel_source),
        expected_task_id=args.expected_task_id,
        expected_mode=args.expected_mode,
    )
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
