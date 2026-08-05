from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_committer_alias_group_guard_sm121a_codegen_20260805"
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_alias_group_artifact_binds_exact_source_and_geometry() -> None:
    summary = _summary()
    contract = summary["compile_contract"]

    assert summary["schema"] == (
        "fr13.fixed32.committer_alias_group_guard.sm121a.codegen.v1"
    )
    assert summary["revision"] == (
        "ea5e32442a68e901a153ba14855708bab247b44e"
    )
    assert summary["source_path"] == (
        "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    )
    source_line = (
        (ARTIFACT / "source_checksums.sha256").read_text().splitlines()[0]
    )
    assert source_line.endswith(
        ":src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    )
    assert summary["source_sha256"] == source_line.split("  ", 1)[0]
    assert contract == {
        "alias_width": 3,
        "bank_rows_fixture": 257,
        "batches": [1, 4],
        "jit_specialization": "mock_tensor_exact_shape_stride_and_alignment",
        "layers": 48,
        "num_stages": 1,
        "num_warps": 4,
        "path_capacity": 16,
        "peer_capacity": 16,
        "physical_rows": 32,
        "target": "sm_121a",
    }


def test_alias_group_codegen_reduces_aggregate_work_without_spills() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["incumbent_owner_sticky"]["builds"][batch]
        candidate = variants["candidate_alias_group_sticky"]["builds"][batch]
        incumbent_work = incumbent["logical_work"]
        candidate_work = candidate["logical_work"]

        assert candidate_work["programs_per_event"] * 3 == incumbent_work[
            "programs_per_event"
        ]
        assert candidate_work["peer_running_row_values"] * 3 == incumbent_work[
            "peer_running_row_values"
        ]
        assert candidate_work["physical_ssi_row_values"] == incumbent_work[
            "physical_ssi_row_values"
        ]
        assert candidate["registers"] <= 32
        assert all(
            candidate[key] == 0
            for key in (
                "stack_bytes",
                "local_bytes",
                "launch_shared_bytes",
                "elf_shared_bytes",
                "ldl",
                "stl",
                "lds",
                "sts",
                "bar",
                "calls",
            )
        )
        assert candidate["global_atomics"] == incumbent["global_atomics"] == 1
        assert candidate["stg"] == incumbent["stg"] == 0

        incumbent_programs = incumbent_work["programs_per_event"]
        candidate_programs = candidate_work["programs_per_event"]
        assert candidate_programs * candidate["static_sass_instructions"] < (
            incumbent_programs * incumbent["static_sass_instructions"]
        )
        assert candidate_programs * candidate["ldg"] < (
            incumbent_programs * incumbent["ldg"]
        )


def test_alias_group_artifact_is_reduced_and_reproducible() -> None:
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
    for name, digest in checksums.items():
        assert (
            hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest()
            == digest
        )
