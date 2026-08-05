from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_b1_gate_b_m128_directgrid_marker_mode_fail_20260805"
)


def _json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text())


def test_failure_preserves_positive_first_comparator_without_qualification() -> None:
    record = _json("comparator_pass.json")
    manifest = _json("manifest.json")

    assert record == {
        "schema": "fr13.fixed32.cutlass_identity_wide256_fullgrid_b1_byte_ab.v1",
        "invocation": 0,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "m": 32,
        "n": 16384,
        "k": 5120,
        "bytes": 1048576,
        "byte_equal": True,
        "mismatch_count": 0,
        "first_mismatch": None,
    }
    assert manifest["status"] == "INCOMPLETE"
    assert manifest["first_comparator"]["byte_equal"] is True
    assert manifest["failure"]["credential_issued"] is False
    assert manifest["claims"]["qualification_complete"] is False
    assert manifest["claims"]["production_eligible"] is False
    assert manifest["claims"]["timing_eligible"] is False


def test_source_manifest_is_bound_to_failed_run_source() -> None:
    manifest = _json("manifest.json")
    source = _json("source_manifest.json")
    source_path = ARTIFACT / manifest["source_identity"]["path"]

    assert source["source_commit"] == manifest["source_commit"]
    assert source["schema"] == manifest["source_identity"]["schema"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        manifest["source_identity"]["sha256"]
    )
    for relative, identity in source["files"].items():
        content = subprocess.check_output(
            ["git", "show", f"{source['source_commit']}:{relative}"], cwd=ROOT
        )
        assert len(content) == identity["bytes"]
        assert hashlib.sha256(content).hexdigest() == identity["sha256"]


def test_artifact_is_sanitized_and_checksum_complete() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "comparator_pass.json",
        "manifest.json",
        "source_manifest.json",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected

    published = "\n".join(
        path.read_text()
        for path in ARTIFACT.iterdir()
        if path.name != "SHA256SUMS"
    )
    for forbidden in (
        "/home/",
        "/tmp/",
        '"prompt":',
        '"response":',
        '"patch":',
        '"container_id":',
        '"pid":',
    ):
        assert forbidden not in published

    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, digest in checksums.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
