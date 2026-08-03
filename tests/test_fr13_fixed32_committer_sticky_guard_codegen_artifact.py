from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_committer_sticky_guard_sm121a_codegen_20260803"
)


def _summary() -> dict[str, object]:
    return json.loads((ARTIFACT / "codegen_summary.json").read_text())


def test_sticky_guard_artifact_binds_exact_source_and_geometry() -> None:
    summary = _summary()
    contract = summary["compile_contract"]

    assert summary["schema"] == (
        "fr13.fixed32.committer_sticky_guard.sm121a.codegen.v1"
    )
    assert summary["revision"] == "0ef914864785fdec62f92f72776a7de0df04cc8a"
    assert summary["source_path"] == (
        "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    )
    source_line = (ARTIFACT / "source_checksums.sha256").read_text().splitlines()[0]
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


def test_sticky_guard_codegen_removes_valid_result_store_without_resources() -> None:
    variants = _summary()["variants"]
    for batch in ("b1", "b4"):
        incumbent = variants["incumbent_vector_result"]["builds"][batch]
        candidate = variants["candidate_sticky_scalar"]["builds"][batch]

        assert candidate["registers"] <= incumbent["registers"]
        assert candidate["ldg"] == incumbent["ldg"] == 8
        assert incumbent["stg"] == 1
        assert candidate["stg"] == 0
        assert incumbent["global_atomics"] == 0
        assert candidate["global_atomics"] == 1
        assert candidate["encoded_sass_instructions"] == incumbent[
            "encoded_sass_instructions"
        ]
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
        incumbent_work = incumbent["logical_work"]
        candidate_work = candidate["logical_work"]
        assert incumbent_work["scalar_reduction_launches_per_event"] == 1
        assert candidate_work["scalar_reduction_launches_per_event"] == 0
        assert candidate_work[
            "source_visible_guard_pipeline_launches_per_event"
        ] == 2
        assert candidate_work["result_bytes_stored_on_valid_event"] == 0
        assert candidate_work["sticky_failure_atomics_on_valid_event"] == 0


def test_sticky_guard_artifact_is_reduced_and_reproducible() -> None:
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
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == digest
