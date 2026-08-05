#!/usr/bin/env python3
"""Verify sanitized fixed32 GDN GQA3 BV16 SM121a evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise SystemExit("CUDA_VISIBLE_DEVICES must be explicitly empty")


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASELINE = "d08e72671a701111179a61ec6125af108be113c0"
CANDIDATE = "4ad046bb5d8658eaa5cdd9d8deddddaf69694da7"
SCHEMA = (
    "fr13.fixed32.gdn_gqa_group3_bv16_source_bound.sm121a.codegen.v1"
)
EXPECTED = {
    "baseline_bv8_base": (8, 108, 1972, 74, 54, None),
    "candidate_bv16_base": (16, 128, 2602, 89, 54, None),
    "baseline_bv8_committer_stack": (8, 118, 2078, 74, 82, 128),
    "candidate_bv16_committer_stack": (16, 112, 2790, 89, 82, 128),
}
FORBIDDEN = {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".llir"}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(name: str) -> dict[str, object]:
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"missing regular artifact file: {name}")
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise RuntimeError(f"artifact JSON is not an object: {name}")
    return value


def verify_summary() -> None:
    summary = load("codegen_summary.json")
    if (
        summary.get("schema") != SCHEMA
        or summary.get("baseline_revision") != BASELINE
        or summary.get("candidate_revision") != CANDIDATE
    ):
        raise RuntimeError("schema or source revisions drifted")
    contract = summary["compile_contract"]
    exact = {
        "target": "sm_121a",
        "batches": [1, 4],
        "physical_rows_per_request": 32,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "selector": "gqa_group3_bv16",
        "candidate_default_off": True,
        "served_reference_block_v": 8,
        "candidate_block_v": 16,
        "gpu_execution": False,
    }
    for key, value in exact.items():
        if contract.get(key) != value:
            raise RuntimeError(f"compile contract drifted: {key}")
    if contract.get("programs_per_48_layer_event") != {
        "baseline_bv8": {"b1": 12288, "b4": 49152},
        "candidate_bv16": {"b1": 6144, "b4": 24576},
        "removed": {"b1": 6144, "b4": 24576},
    }:
        raise RuntimeError("event program-count math drifted")
    variants = summary.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(EXPECTED):
        raise RuntimeError("variant set drifted")
    for name, metrics in EXPECTED.items():
        block_v, registers, static_sass, ldg, stg, maxnreg = metrics
        for batch, programs in ((1, 6144), (4, 24576)):
            row = variants[name]["builds"][f"b{batch}"]
            expected_programs = programs if block_v == 16 else programs * 2
            if (
                row.get("block_v") != block_v
                or row.get("grid") != [16, 128 // block_v, batch]
                or row.get("programs_per_48_layer_event") != expected_programs
                or row.get("registers_per_thread") != registers
                or row.get("static_sass_instructions") != static_sass
                or row.get("ldg") != ldg
                or row.get("stg") != stg
                or row.get("resolved_maxnreg") != maxnreg
                or row.get("stack_bytes_per_thread") != 0
                or row.get("local_bytes_per_thread") != 0
                or row.get("ldl") != 0
                or row.get("stl") != 0
                or row.get("calls") != 0
                or row.get("gpu_execution") is not False
            ):
                raise RuntimeError(f"resource evidence drifted: {name}/b{batch}")


def verify_manifest(name: str) -> None:
    for line in (ROOT / name).read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        if name == "source_checksums.sha256" and relative.startswith(
            ("src/", "scripts/")
        ):
            raw = subprocess.run(
                ["git", "-C", str(REPO), "show", f"{CANDIDATE}:{relative}"],
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        else:
            path = REPO / relative if name == "source_checksums.sha256" else ROOT / relative
            raw = path.read_bytes()
        if sha256(raw) != expected:
            raise RuntimeError(f"checksum drifted: {relative}")


def main() -> None:
    verify_summary()
    verification = load("verification.json")
    if verification != {
        "builds_per_compile": 8,
        "compile_count": 2,
        "fresh_cache_byte_identity": True,
        "gpu_execution": False,
        "performance_promotion": False,
        "status": "PASS",
    }:
        raise RuntimeError("verification record drifted")
    if any(path.suffix in FORBIDDEN for path in ROOT.rglob("*")):
        raise RuntimeError("raw compiler output entered sanitized artifact")
    verify_manifest("source_checksums.sha256")
    verify_manifest("SHA256SUMS")
    print("PASS: exact BV16 source-bound SM121a evidence is verified")


if __name__ == "__main__":
    main()
