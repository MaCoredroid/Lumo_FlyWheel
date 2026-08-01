#!/usr/bin/env python3
"""Verify the compile-only projection row-cover binary and reduced evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fr13_cutlass_wave_binary as binary
import fr13_projection_rowcover_b1_pass as b1_qualification


SCHEMA = "fr13.fixed32.projection_rowcover.static_qualification.v1"
EXPECTED_RESOURCE_ROWS = {
    ("b1_static_persistent", "fp16"): (128, 32, 128, 168, 0, 0, 1024, 2688),
    ("b1_static_persistent", "bf16"): (128, 32, 128, 168, 0, 0, 1024, 2688),
    ("b4_persistent_m128", "fp16"): (128, 128, 128, 168, 0, 0, 1024, 2560),
    ("b4_persistent_m128", "bf16"): (128, 128, 128, 168, 0, 0, 1024, 2560),
}


class StaticQualificationError(ValueError):
    """The compile-only binary or reduced evidence is inconsistent."""


def _parse_resources(path: Path) -> dict[tuple[str, str], tuple[int, ...]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    parsed: dict[tuple[str, str], tuple[int, ...]] = {}
    for row in rows:
        key = (row["candidate"], row["output_type"])
        if key in parsed:
            raise StaticQualificationError(f"duplicate resource row: {key!r}")
        parsed[key] = tuple(
            int(row[name])
            for name in (
                "tile_m",
                "tile_n",
                "tile_k",
                "registers",
                "stack_bytes",
                "local_bytes",
                "shared_bytes",
                "constant_0_bytes",
            )
        )
    return parsed


def _parse_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise StaticQualificationError("stock-equivalence evidence is malformed")
        values[key] = value
    return values


def qualify(
    candidate_so: Path,
    patch_source: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    b1 = binary.verify_candidate(
        candidate_so, "static_persistent_stocktile_byte_ab"
    )
    b4 = binary.verify_candidate(candidate_so, "persistent_b4_m128_byte_ab")
    if (b1["sha256"], b1["bytes"], b1["candidate_family"]) != (
        b4["sha256"],
        b4["bytes"],
        b4["candidate_family"],
    ):
        raise StaticQualificationError("B1 and B4 selectors do not share one binary")
    patch_sha256 = b1_qualification.sha256_file(patch_source)
    if patch_sha256 != b1_qualification.PATCH_SOURCE_SHA256:
        raise StaticQualificationError("projection row-cover patch source drifted")

    resources = _parse_resources(evidence_dir / "kernel_resources.tsv")
    if resources != EXPECTED_RESOURCE_ROWS:
        raise StaticQualificationError(
            "candidate kernel resources differ from contract"
        )
    stock = _parse_key_values(evidence_dir / "stock_equivalence.txt")
    required_stock = {
        "status": "pass",
        "candidate_binary_sha256": str(b1["sha256"]),
        "reference_stock_record_count": "6",
        "candidate_stock_record_count": "6",
        "matched_stock_record_count": "6",
        "missing_stock_record_count": "0",
        "strong_dynamic_reference_count": "873",
        "strong_dynamic_candidate_count": "873",
        "strong_dynamic_comparison": "exact",
    }
    for key, expected in required_stock.items():
        if stock.get(key) != expected:
            raise StaticQualificationError(
                f"stock-equivalence {key} mismatch: {stock.get(key)!r} != {expected!r}"
            )
    candidate_record = json.loads(
        (evidence_dir / "candidate.json").read_text(encoding="ascii")
    )
    expected_candidate_fields = {
        "source": {
            "patch_sha256": b1_qualification.PATCH_SOURCE_SHA256,
            "patched_dispatch_sha256": b1_qualification.PATCHED_DISPATCH_SHA256,
        },
        "build": {
            "binary_sha256": b1["sha256"],
            "binary_bytes": b1["bytes"],
            "binary_mode": "0555",
        },
    }
    for section, expected_values in expected_candidate_fields.items():
        actual = candidate_record.get(section)
        if not isinstance(actual, dict):
            raise StaticQualificationError(f"candidate evidence lacks {section}")
        for key, expected in expected_values.items():
            if actual.get(key) != expected:
                raise StaticQualificationError(
                    f"candidate evidence {section}.{key} mismatch"
                )

    return {
        "schema": SCHEMA,
        "status": "pass",
        "acceptance_valid": False,
        "timing_eligible": False,
        "performance_claim": False,
        "candidate_sha256": b1["sha256"],
        "candidate_bytes": b1["bytes"],
        "candidate_family": b1["candidate_family"],
        "patch_source_sha256": patch_sha256,
        "patched_dispatch_sha256": b1_qualification.PATCHED_DISPATCH_SHA256,
        "b1_selector": "static_persistent_stocktile",
        "b1_fixed_rows": 32,
        "b4_selector": "persistent_b4_m128",
        "b4_fixed_rows": 128,
        "resource_records": len(resources),
        "zero_stack_records": sum(row[4] == 0 for row in resources.values()),
        "zero_local_records": sum(row[5] == 0 for row in resources.values()),
        "stock_resource_records_matched": int(stock["matched_stock_record_count"]),
        "strong_dynamic_exports_matched": int(stock["strong_dynamic_candidate_count"]),
        "requires_fresh_b1_k64_byte_gate": True,
        "requires_fresh_b4_k64_exact4_byte_gate": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument(
        "--patch-source", type=Path, default=b1_qualification.PATCH_SOURCE
    )
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = qualify(args.candidate_so, args.patch_source, args.evidence_dir)
    encoded = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="ascii")
    print(json.dumps(payload, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
