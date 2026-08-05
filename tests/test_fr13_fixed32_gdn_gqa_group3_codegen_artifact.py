from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_gdn_gqa_group3_node_domain_sm121a_20260805"
)
BASELINE_REVISION = "9091ddae2046f42fc5e754f976c3493a033785ac"
CANDIDATE_REVISION = "8c85135cb6092f01230d93c55b1c6f3fcf7336f3"
PROFILE_PAIRS = (
    (
        "baseline_gqa_group3_base",
        "candidate_node_domain_base",
        120,
        116,
        2174,
        2052,
        54,
        None,
    ),
    (
        "baseline_gqa_group3_committer_stack",
        "candidate_node_domain_committer_stack",
        126,
        122,
        2280,
        2155,
        82,
        128,
    ),
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def _build(summary: dict[str, object], variant: str):
    return summary["variants"][variant]["builds"]["b4"]


def test_exact_b4_profiles_are_spill_free_and_launch_viable() -> None:
    summary = _summary()
    assert summary["schema"] == (
        "fr13.fixed32.gdn_gqa_group3_node_domain.sm121a.codegen.v1"
    )
    assert summary["baseline_revision"] == BASELINE_REVISION
    assert summary["candidate_revision"] == CANDIDATE_REVISION
    assert summary["compile_contract"]["target"] == "sm_121a"
    assert summary["compile_contract"]["batches"] == [4]
    assert summary["compile_contract"]["physical_rows_per_request"] == 32
    assert summary["compile_contract"]["candidate_trust_fixed32_node_domain"]
    for baseline_name, candidate_name, *_metrics in PROFILE_PAIRS:
        baseline = _build(summary, baseline_name)
        candidate = _build(summary, candidate_name)
        assert baseline["grid"] == [16, 16, 4]
        assert candidate["grid"] == [16, 16, 4]
        assert baseline["trust_fixed32_node_domain"] is False
        assert candidate["trust_fixed32_node_domain"] is True
        for row in (baseline, candidate):
            assert row["threads_per_cta"] == 256
            assert row["programs_per_layer_event"] == 1024
            assert row["programs_per_48_layer_event"] == 49152
            assert row["launch_shared_bytes_per_cta"] == 16
            assert row["elf_shared_bytes_per_cta"] == 1024
            assert row["stack_bytes_per_thread"] == 0
            assert row["local_bytes_per_thread"] == 0
            assert row["ldl"] == 0
            assert row["stl"] == 0
            assert row["calls"] == 0
            assert row["gpu_execution"] is False


def test_node_domain_specialization_reduces_static_resources() -> None:
    summary = _summary()
    for (
        baseline_name,
        candidate_name,
        baseline_registers,
        candidate_registers,
        baseline_static,
        candidate_static,
        stg,
        maxnreg,
    ) in PROFILE_PAIRS:
        baseline = _build(summary, baseline_name)
        candidate = _build(summary, candidate_name)
        assert baseline["registers_per_thread"] == baseline_registers
        assert candidate["registers_per_thread"] == candidate_registers
        assert candidate_registers == baseline_registers - 4
        assert baseline["static_sass_instructions"] == baseline_static
        assert candidate["static_sass_instructions"] == candidate_static
        assert candidate_static < baseline_static
        assert baseline["ldg"] == candidate["ldg"] == 85
        assert baseline["stg"] == candidate["stg"] == stg
        assert baseline["resolved_maxnreg"] == maxnreg
        assert candidate["resolved_maxnreg"] == maxnreg
        assert baseline["cubin_sha256"] != candidate["cubin_sha256"]


def test_artifact_source_hashes_and_sanitized_scope() -> None:
    for line in (ARTIFACT / "source_checksums.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    forbidden = {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".llir"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    readme = (ARTIFACT / "README.md").read_text()
    assert "offline SM121a codegen gate PASS" in readme
    assert "not performance-promoted" in readme
    assert "No GPU kernel was launched" in readme
    assert "SWE-Verified B4 byte-equivalence gate" in readme


def test_artifact_package_checksums_match() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    files = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(entries) == files
    for relative, expected in entries.items():
        observed = hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest()
        assert observed == expected


def test_verification_is_static_and_not_performance_promotion() -> None:
    verification = json.loads((ARTIFACT / "verification.json").read_text())
    assert verification["status"] == "PASS"
    assert verification["fresh_cache_byte_identity"] is True
    assert verification["global_memory_operations_unchanged"] is True
    assert verification["performance_promotion"] is False
    assert verification["gpu_execution"] is False
