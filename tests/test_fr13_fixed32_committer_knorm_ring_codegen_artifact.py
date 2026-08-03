from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_committer_knorm_ring_sm121a_codegen_20260803"
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def _count(build: dict[str, object], operation: str) -> int:
    return int(build["base_operations"].get(operation, 0))


def _full_count(build: dict[str, object], operation: str) -> int:
    return int(build["full_operations"].get(operation, 0))


def test_knorm_artifact_binds_exact_source_and_geometry() -> None:
    summary = _summary()
    contract = summary["compile_contract"]

    assert summary["schema"] == (
        "fr13.fixed32.committer_knorm_ring.sm121a.codegen.v1"
    )
    assert summary["revision"] == (
        "b2b4ab6f5ec4ec1f7ac6b5606b711ef2c1f68d37"
    )
    assert summary["parent_revision"] == (
        "178193bd5226d090fa52d5052e93a0f3a6bc0e06"
    )
    source_line = (ARTIFACT / "source_checksums.sha256").read_text().splitlines()[0]
    assert source_line.endswith(
        ":src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    )
    assert summary["source_sha256"] == source_line.split("  ", 1)[0]
    assert contract["target"] == "sm_121a"
    assert contract["batches"] == [1, 4]
    assert contract["physical_rows"] == 32
    assert contract["key_heads"] == 16
    assert contract["value_heads"] == 48
    assert contract["dim_k"] == contract["dim_v"] == 128
    assert contract["producer_block_v"] == 8
    assert contract["committer_block_v"] == 128
    assert contract["num_warps"] == 8


def test_knorm_default_off_sass_is_parent_identical() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        for parent_label, current_label in (
            (
                "producer_parent_incumbent",
                "producer_current_incumbent",
            ),
            (
                "committer_parent_incumbent",
                "committer_current_incumbent",
            ),
        ):
            parent = variants[parent_label]["builds"][batch]
            current = variants[current_label]["builds"][batch]
            assert parent["sass_sha256"] == current["sass_sha256"]
            assert parent["base_operations"] == current["base_operations"]
            assert parent["registers"] == current["registers"]


def test_knorm_producer_reuses_existing_reduction_without_spill() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["producer_current_incumbent"]["builds"][batch]
        candidate = variants["producer_candidate_knorm_export"]["builds"][batch]

        assert candidate["registers"] == incumbent["registers"] == 80
        assert _full_count(candidate, "MUFU.RSQ") == _full_count(
            incumbent, "MUFU.RSQ"
        )
        for operation in ("LDG", "MUFU", "SHFL", "BAR", "LDS", "STS"):
            assert _count(candidate, operation) == _count(incumbent, operation)
        assert _count(candidate, "STG") > _count(incumbent, "STG")
        assert candidate["logical_work"]["key_norm_reductions_added"] == 0
        assert all(
            candidate[key] == 0 for key in ("stack_bytes", "local_bytes")
        )
        assert all(
            _count(candidate, operation) == 0
            for operation in ("LDL", "STL", "CALL")
        )


def test_knorm_committer_removes_reduction_and_shared_roundtrip() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["committer_current_incumbent"]["builds"][batch]
        candidate = variants["committer_candidate_knorm_reuse"]["builds"][batch]

        assert _full_count(incumbent, "MUFU.RSQ") == 1
        assert _full_count(candidate, "MUFU.RSQ") == 0
        assert _count(incumbent, "SHFL") == 87
        assert _count(candidate, "SHFL") == 80
        assert _count(incumbent, "BAR") == 3
        assert _count(candidate, "BAR") == 0
        assert _count(candidate, "LDS") == _count(candidate, "STS") == 0
        assert candidate["launch_shared_bytes"] == 0
        assert candidate["registers"] < 200
        assert all(
            candidate[key] == 0 for key in ("stack_bytes", "local_bytes")
        )
        for depth in ("accepted_0", "accepted_4", "accepted_11"):
            before = incumbent["logical_work"]["dynamic_step_census"][depth]
            after = candidate["logical_work"]["dynamic_step_census"][depth]
            assert after["key_norm_reductions"] == 0
            assert after["key_norm_reductions_removed"] == before[
                "key_norm_reductions"
            ]
            assert after["key_norm_scalar_loads"] == before[
                "key_norm_reductions"
            ]


def test_knorm_artifact_is_reduced_and_reproducible() -> None:
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
