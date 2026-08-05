from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO / "results" / "fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805"
)
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu"
BUILDER = REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m1_shuffle_r64_u8.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reproducibility_record_binds_exact_source_and_builder() -> None:
    payload = json.loads((ARTIFACT / "reproducibility.json").read_text())
    assert payload["status"] == "REPRODUCIBLE_LINKED_BUILD_UNQUALIFIED"
    assert payload["byte_identical_fresh_builds"] is True
    assert payload["gpu_runtime_used"] is False
    assert payload["real_task_correctness"] is False
    assert payload["runtime_wired"] is False
    assert payload["source_sha256"] == sha256(SOURCE)
    assert payload["build_script_sha256"] == sha256(BUILDER)
    assert payload["binary"]["primary_sha256"] == payload["binary"]["rebuild_sha256"]
    assert payload["cubin"]["primary_sha256"] == payload["cubin"]["rebuild_sha256"]


def test_artifact_records_sm121a_resources_without_shipping_binary() -> None:
    resources = (ARTIFACT / "resource_usage.txt").read_text(encoding="ascii")
    cubins = (ARTIFACT / "cubin_list.txt").read_text(encoding="ascii")
    assert "arch = sm_121a" in resources
    assert "REG:29 STACK:0 SHARED:0 LOCAL:0" in resources
    assert "sm_121a.cubin" in cubins
    assert not list(ARTIFACT.glob("*.so"))
    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        checksums[relative] = digest
    expected = {
        str(path.relative_to(REPO))
        for path in ARTIFACT.iterdir()
        if path.name != "SHA256SUMS"
    }
    assert set(checksums) == expected
    assert all(sha256(REPO / path) == digest for path, digest in checksums.items())
