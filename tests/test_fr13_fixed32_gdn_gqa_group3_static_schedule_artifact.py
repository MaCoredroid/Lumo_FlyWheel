from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_gdn_gqa_group3_static_schedule_sm121a_20260805"
)
BASELINE_REVISION = "6c28fc58992e495bd8d4c8640370cc82f17316ee"
CANDIDATE_REVISION = "a5174ed5e8ac2d5768a4a9e0fda16786c564e40a"
PROFILE_PAIRS = (
    (
        "baseline_node_domain_base",
        "candidate_static_schedule_base",
        116,
        116,
        2052,
        2012,
        54,
        None,
    ),
    (
        "baseline_node_domain_committer_stack",
        "candidate_static_schedule_committer_stack",
        122,
        118,
        2155,
        2119,
        82,
        128,
    ),
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_exact_b1_b4_profiles_remove_descriptor_loads_without_spills() -> None:
    summary = _summary()
    assert summary["schema"] == (
        "fr13.fixed32.gdn_gqa_group3_static_schedule.sm121a.codegen.v1"
    )
    assert summary["baseline_revision"] == BASELINE_REVISION
    assert summary["candidate_revision"] == CANDIDATE_REVISION
    contract = summary["compile_contract"]
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows_per_request"] == 32
    assert contract["candidate_static_physical32_schedule"] is True
    assert contract["device_descriptor_pointer_args_removed"] == 5
    assert contract["device_descriptor_loads_removed_per_cta"] == 59
    assert contract["device_descriptor_loads_removed_per_48_layer_event"] == {
        "b1": 724_992,
        "b4": 2_899_968,
    }

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
        for batch in (1, 4):
            label = f"b{batch}"
            baseline = summary["variants"][baseline_name]["builds"][label]
            candidate = summary["variants"][candidate_name]["builds"][label]
            assert baseline["registers_per_thread"] == baseline_registers
            assert candidate["registers_per_thread"] == candidate_registers
            assert baseline["static_sass_instructions"] == baseline_static
            assert candidate["static_sass_instructions"] == candidate_static
            assert baseline["ldg"] == 85
            assert candidate["ldg"] == 74
            assert baseline["stg"] == candidate["stg"] == stg
            assert baseline["resolved_maxnreg"] == maxnreg
            assert candidate["resolved_maxnreg"] == maxnreg
            for row in (baseline, candidate):
                assert row["stack_bytes_per_thread"] == 0
                assert row["local_bytes_per_thread"] == 0
                assert row["ldl"] == row["stl"] == row["calls"] == 0
                assert row["gpu_execution"] is False


def test_artifact_is_reproducible_sanitized_static_evidence() -> None:
    verification = json.loads((ARTIFACT / "verification.json").read_text())
    assert verification["status"] == "PASS"
    assert verification["builds_verified"] == 8
    assert verification["fresh_cache_byte_identity"] is True
    assert verification["performance_promotion"] is False
    assert verification["gpu_execution"] is False

    forbidden = {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".llir"}
    assert not any(path.suffix in forbidden for path in ARTIFACT.rglob("*"))
    readme = (ARTIFACT / "README.md").read_text()
    assert "offline SM121a codegen/resource gate PASS" in readme
    assert "59 executed descriptor loads per CTA" in readme
    assert "not performance-promoted" in readme
    assert "No GPU kernel was launched" in readme


def test_artifact_checksums_match() -> None:
    for manifest in ("source_checksums.sha256", "SHA256SUMS"):
        for line in (ARTIFACT / manifest).read_text().splitlines():
            expected, relative = line.split("  ", 1)
            source = (
                ROOT / relative
                if manifest == "source_checksums.sha256"
                else ARTIFACT / relative
            )
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
