#!/usr/bin/env python3
"""Validate the source-bound direct-nodegroup8 B1 readiness artifact."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = Path(__file__).resolve().parent
SOURCE_COMMIT = "89f785fb560c2b7fefb7fa5a61171b5b0316fc4c"
SOURCE_MANIFEST_SHA256 = (
    "a98cd762feca502b3e33bba6b06b9d58c3731c55f6aa4eafb878dbdc07bfa494"
)
HOST_READINESS_SHA256 = (
    "e232c755db0510a34325b45e271235d13eaa103d3a292cfa0ce79d4c08763a5b"
)


def _raw(name: str) -> bytes:
    return (ARTIFACT / name).read_bytes()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    manifest_raw = _raw("source_manifest.json")
    readiness_raw = _raw("host_readiness.json")
    assert _sha256(manifest_raw) == SOURCE_MANIFEST_SHA256
    assert _sha256(readiness_raw) == HOST_READINESS_SHA256
    manifest = json.loads(manifest_raw.decode("ascii"))
    readiness = json.loads(readiness_raw.decode("ascii"))
    contract = json.loads(_raw("selector_contract.json").decode("ascii"))

    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["candidate"] == (
        "fixed32_sfwd_conv_postprep_nodegroup8_direct_v1"
    )
    assert readiness["source_commit"] == SOURCE_COMMIT
    assert readiness["upstream_commit"] == SOURCE_COMMIT
    assert readiness["source_manifest_sha256"] == SOURCE_MANIFEST_SHA256
    assert readiness["direct_nodegroup8"] is True
    assert readiness["batch_size"] == 1
    assert readiness["physical_rows_per_request"] == 32
    assert readiness["draft_vocab_root"] == 1
    assert readiness["draft_vocab_k"] == 65536
    assert readiness["programs_per_request"] == 164
    assert readiness["reference_always_served"] is True
    assert readiness["candidate_returned"] is False
    assert readiness["gpu_or_docker_used"] is False
    assert readiness["launched"] is False
    assert readiness["timing_eligible"] is False
    assert readiness["floor_acceptance_eligible"] is False
    assert contract["embedded_programs_per_request"] == 160
    assert contract["standalone_programs_per_request"] == 164

    for relative, expected in manifest["files"].items():
        committed = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{SOURCE_COMMIT}:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        assert len(committed) == expected["bytes"], relative
        assert _sha256(committed) == expected["sha256"], relative

    checksums = (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    expected_names = {
        "README.md",
        "host_readiness.json",
        "selector_contract.json",
        "source_manifest.json",
        "validate_artifact.py",
        "verification.txt",
    }
    observed_names = set()
    for line in checksums:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        assert match is not None, line
        digest, name = match.groups()
        observed_names.add(name)
        assert _sha256(_raw(name)) == digest, name
    assert observed_names == expected_names
    print("artifact validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
