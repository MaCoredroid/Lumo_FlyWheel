#!/usr/bin/env python3
"""Verify the reduced fixed32 CFWD sparse-decision artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    expected_files = {
        "README.md",
        "SHA256SUMS",
        "codegen_summary.json",
        "codegen_summary.tsv",
        "manifest.json",
        "math_contract.json",
        "test_results.txt",
        "verify_artifact.py",
    }
    require(
        {path.name for path in ROOT.iterdir()} == expected_files,
        "artifact file set drifted",
    )
    for line in (ROOT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        require(name in expected_files - {"SHA256SUMS"}, f"unknown checksum {name}")
        require(
            hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest,
            f"checksum mismatch: {name}",
        )

    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="ascii"))
    summary = json.loads(
        (ROOT / "codegen_summary.json").read_text(encoding="ascii")
    )
    math_contract = json.loads(
        (ROOT / "math_contract.json").read_text(encoding="ascii")
    )
    require(
        manifest["schema"]
        == "fr13.fixed32.cfwd_sparse_decisions.codegen_artifact.v1",
        "manifest schema drifted",
    )
    require(manifest["status"] == "offline_source_codegen_pass", "bad status")
    for field in (
        "acceptance_valid",
        "default_enabled",
        "docker_used",
        "floor_acceptance_eligible",
        "gpu_kernel_launched",
        "gpu_queried",
        "production_enabled",
        "real_swe_verified_task_run",
        "synthetic_or_probe_timing_run",
        "timing_eligible",
    ):
        require(manifest[field] is False, f"{field} must be false")
    require(summary["source_sha256"] == manifest["candidate_source"]["sha256"], "source drift")
    require(summary["architecture"] == "sm_121a", "architecture drift")
    require(math_contract["candidate"] == manifest["candidate"], "math identity drift")
    require(len(summary["kernels"]) == 4, "kernel count drift")
    require(
        max(kernel["registers"] for kernel in summary["kernels"]) == 42,
        "register ceiling drift",
    )
    for kernel in summary["kernels"]:
        require(kernel["spill_loads"] == 0, "spill load present")
        require(kernel["spill_stores"] == 0, "spill store present")
        require(kernel["stack_bytes"] == 0, "stack frame present")
        require(kernel["local_bytes"] == 0, "local memory present")
        require(kernel["calls"] == 0, "device call present")
    print("PASS fr13 fixed32 CFWD sparse-decision artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
