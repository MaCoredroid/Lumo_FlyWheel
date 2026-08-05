from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARTIFACT = (
    REPO
    / "results"
    / "fr13_fixed32_dfwd_k64_m4_r64_u8_linked_build_20260805"
)
SOURCE = REPO / "csrc" / "fr13_bf16_gemvx_k64_m4_shuffle_r64_u8.cu"
BUILDER = (
    REPO / "scripts" / "fr13_build_bf16_gemvx_k64_m4_shuffle_r64_u8.py"
)
CHECKER = (
    REPO
    / "scripts"
    / "fr13_check_bf16_gemvx_k64_m4_shuffle_r64_u8_codegen.py"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_build_attestation_binds_source_builder_and_binary() -> None:
    payload = json.loads((ARTIFACT / "build_attestation.json").read_text())
    assert payload["schema"] == (
        "fr13.fixed32.dfwd_k64_m4_r64_u8_sm121a_canonical_build.v1"
    )
    assert payload["status"] == (
        "REPRODUCIBLE_CANONICAL_LINKED_BUILD_UNQUALIFIED"
    )
    for key in (
        "byte_equality_claim",
        "docker_used",
        "gpu_runtime_used",
        "performance_measurement",
        "production_default_enabled",
        "real_task_correctness",
        "runtime_wired",
    ):
        assert type(payload[key]) is bool
    assert payload["source"]["sha256"] == sha256(SOURCE)
    assert payload["builder"]["sha256"] == sha256(BUILDER)
    assert payload["checker"]["sha256"] == sha256(CHECKER)
    assert payload["binary"] == {
        "bytes": 134320,
        "mode": "0555",
        "path": (
            "/home/mark/shared/fr13_dfwd_m4_u8_linked_build_"
            "bb8a4a8a2_20260805/canonical-primary-bin/"
            "fr13_bf16_k64_m4_r64_u8.abi3.so"
        ),
        "sha256": (
            "6cb24782495ff1c1457ebbf9cbcfcd6ca7b372378d3b435f80054688432a365f"
        ),
    }
    assert payload["rebuild_binary"]["sha256"] == payload["binary"]["sha256"]
    assert payload["cubin"]["sha256"] == (
        "952395db481f7c1b8d8c631789d961f41c6ec8d3cbb85c80b9ee5f1b2371a4e1"
    )
    assert payload["registered_operation"] == (
        "fr13_bf16_k64_head::gemvx_m4_shuffle_r64_u8_out"
    )


def test_reproducibility_record_scopes_raw_variance_and_proves_outputs() -> None:
    payload = json.loads((ARTIFACT / "reproducibility.json").read_text())
    assert payload["status"] == (
        "REPRODUCIBLE_CANONICAL_LINKED_BUILD_UNQUALIFIED"
    )
    assert payload["canonical_binary"]["byte_identical_fresh_builds"] is True
    assert (
        payload["canonical_binary"]["primary_sha256"]
        == payload["canonical_binary"]["rebuild_sha256"]
    )
    assert payload["cubin"]["byte_identical_fresh_builds"] is True
    assert payload["cubin"]["primary_sha256"] == payload["cubin"]["rebuild_sha256"]
    raw = payload["raw_binary"]
    assert raw["differing_byte_count"] == 6
    assert raw["difference_section"] == ".strtab"
    assert raw["loadable_bytes_identical"] is True
    assert raw["build_id_primary"] == raw["build_id_rebuild"]
    assert raw["primary_sha256"] != raw["rebuild_sha256"]
    assert payload["canonicalization"]["output_load_registration_primary"] is True
    assert payload["canonicalization"]["output_load_registration_rebuild"] is True
    assert payload["gpu_runtime_used"] is False
    assert payload["docker_used"] is False
    assert payload["real_task_correctness"] is False
    assert payload["runtime_wired"] is False


def test_linked_cubin_passes_exact_b4_static_contract() -> None:
    audit = json.loads((ARTIFACT / "linked_codegen_audit.json").read_text())
    assert audit["status"] == "STATIC_CODEGEN_PASS_UNQUALIFIED"
    assert audit["candidate_static_instructions"] == 200
    assert audit["candidate_steady_loop_instructions"] == 119
    assert audit["candidate_resources"] == {
        "constant0_bytes": 928,
        "local_bytes_per_thread": 0,
        "registers_per_thread": 56,
        "shared_bytes_per_cta": 0,
        "stack_bytes_per_thread": 0,
    }
    resources = (ARTIFACT / "resource_usage.txt").read_text(encoding="ascii")
    assert "arch = sm_121a" in resources
    assert "REG:56 STACK:0 SHARED:0 LOCAL:0" in resources
    assert not list(ARTIFACT.glob("*.so"))


def test_readme_pins_real_exact4_five_site_gate_boundary() -> None:
    readme = (ARTIFACT / "README.md").read_text(encoding="ascii")
    for requirement in (
        "FR13_DRAFT_HEAD_M4_R64_U8_*",
        "root and MTP depths 1 through 4",
        "view(torch.int16)",
        "config/fr13_fixed32/subset_b4_four.json",
        "BSIZE=4",
        "CONC=4",
        "FULL_AND_PIECEWISE",
        "always return the stock logits object",
        "no timing or acceptance claim",
    ):
        assert requirement in readme


def test_artifact_sha256_inventory_is_complete() -> None:
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
