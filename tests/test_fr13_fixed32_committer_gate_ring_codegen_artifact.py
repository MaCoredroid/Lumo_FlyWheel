from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_committer_gate_ring_sm121a_codegen_20260803"
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def _count(build: dict[str, object], operation: str) -> int:
    return int(build["base_operations"].get(operation, 0))


def _full_count(build: dict[str, object], operation: str) -> int:
    return int(build["full_operations"].get(operation, 0))


def test_gate_artifact_binds_exact_source_and_geometry() -> None:
    summary = _summary()
    contract = summary["compile_contract"]

    assert summary["schema"] == (
        "fr13.fixed32.committer_gate_ring.sm121a.codegen.v1"
    )
    assert summary["revision"] == (
        "5700ddaf3ff51e0b8dba0d571069ba0d8c158ce6"
    )
    assert summary["parent_revision"] == (
        "12918adaa869d1c88e1424483a189142571406ae"
    )
    source_line = (ARTIFACT / "source_checksums.sha256").read_text().splitlines()[0]
    assert source_line.endswith(
        "  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    )
    assert summary["source_sha256"] == source_line.split("  ", 1)[0]
    assert contract["target"] == "sm_121a"
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows"] == 32
    assert contract["key_heads"] == 16
    assert contract["value_heads"] == 48
    assert contract["dim_k"] == contract["dim_v"] == 128
    assert contract["producer_block_v"] == 8
    assert contract["producer_candidate_maxnreg"] == 80
    assert contract["committer_block_v"] == 128
    assert contract["num_warps"] == 8


def test_gate_knorm_only_sass_is_parent_identical() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        for parent_label, current_label in (
            ("producer_parent_knorm", "producer_current_knorm"),
            ("committer_parent_knorm", "committer_current_knorm"),
        ):
            parent = variants[parent_label]["builds"][batch]
            current = variants[current_label]["builds"][batch]
            assert parent["sass_sha256"] == current["sass_sha256"]
            assert parent["base_operations"] == current["base_operations"]
            assert parent["registers"] == current["registers"]


def test_gate_producer_reuses_existing_math_without_spill() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["producer_current_knorm"]["builds"][batch]
        candidate = variants["producer_candidate_gate_export"]["builds"][batch]

        assert candidate["registers"] == incumbent["registers"] == 80
        for operation in (
            "LDG",
            "MUFU",
            "FADD",
            "FFMA",
            "FMUL",
            "FSEL",
            "SHFL",
            "BAR",
            "LDS",
            "STS",
        ):
            assert _count(candidate, operation) == _count(incumbent, operation)
        for operation in ("MUFU.EX2", "MUFU.RCP", "MUFU.RSQ"):
            assert _full_count(candidate, operation) == _full_count(
                incumbent, operation
            )
        assert _count(candidate, "STG") > _count(incumbent, "STG")
        assert candidate["logical_work"]["gate_nonlinear_evaluations_added"] == 0
        assert all(
            candidate[key] == 0 for key in ("stack_bytes", "local_bytes")
        )
        assert all(
            _count(candidate, operation) == 0
            for operation in ("LDL", "STL", "CALL")
        )


def test_gate_committer_removes_live_step_nonlinears() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["committer_current_knorm"]["builds"][batch]
        candidate = variants["committer_candidate_gate_reuse"]["builds"][batch]

        assert _full_count(incumbent, "MUFU.EX2") == 3
        assert _full_count(candidate, "MUFU.EX2") == 1
        assert _full_count(incumbent, "MUFU.RCP") == 1
        assert _full_count(candidate, "MUFU.RCP") == 0
        assert candidate["registers"] == 167
        assert candidate["registers"] < incumbent["registers"]
        assert candidate["static_sass_instructions"] < incumbent[
            "static_sass_instructions"
        ]
        assert _count(candidate, "LDG") == 40
        assert _count(incumbent, "LDG") == 42
        assert all(
            candidate[key] == 0 for key in ("stack_bytes", "local_bytes")
        )
        for depth in ("accepted_0", "accepted_4", "accepted_11"):
            before = incumbent["logical_work"]["dynamic_step_census"][depth]
            after = candidate["logical_work"]["dynamic_step_census"][depth]
            assert after["gate_nonlinear_sets"] == 0
            assert after["gate_nonlinear_sets_removed"] == before[
                "gate_nonlinear_sets"
            ]
            assert after["gate_scalar_loads"] == 2 * before[
                "gate_nonlinear_sets"
            ]


def test_gate_artifact_is_reduced_and_reproducible() -> None:
    expected = {
        "README.md",
        "SHA256SUMS",
        "codegen_summary.json",
        "offline_codegen_audit.py",
        "source_checksums.sha256",
        "test_results.txt",
        "verification.md",
        "verify_codegen_outputs.py",
    }
    assert {path.name for path in ARTIFACT.iterdir()} == expected
    assert not any(
        path.suffix in {".cubin", ".ptx", ".sass", ".log"}
        for path in ARTIFACT.rglob("*")
    )

    checksums = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    assert set(checksums) == expected - {"SHA256SUMS"}
    for name, expected_digest in checksums.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == (
            expected_digest
        )
