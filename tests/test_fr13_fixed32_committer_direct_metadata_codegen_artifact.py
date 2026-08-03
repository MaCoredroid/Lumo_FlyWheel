from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_committer_direct_metadata_sm121a_codegen_20260803"
)


def test_artifact_is_reduced_offline_sm121a_evidence() -> None:
    files = {path.name for path in ARTIFACT.iterdir() if path.is_file()}
    assert {
        "README.md",
        "codegen_summary.json",
        "offline_codegen_audit.py",
        "source_checksums.sha256",
        "test_results.txt",
        "verification.md",
        "verify_codegen_outputs.py",
    } <= files
    assert not any(
        path.suffix in {".cubin", ".ptx", ".sass", ".jsonl", ".log"}
        for path in ARTIFACT.rglob("*")
        if path.is_file()
    )
    compiler = (ARTIFACT / "offline_codegen_audit.py").read_text()
    verifier = (ARTIFACT / "verify_codegen_outputs.py").read_text()
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in compiler
    assert 'GPUTarget("cuda", 121, 32)' in compiler
    assert "MockTensor" in compiler
    assert "nvdisasm" in verifier
    assert "fresh-cache summary differs" in verifier


def test_summary_keeps_candidate_on_the_resource_pareto_frontier() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["revision"] == "0e2f3b940ee7076e7818da4e048206a978236f04"
    assert summary["compile_contract"] == {
        "batches": [1, 4],
        "block_channels": 1024,
        "channels": 10240,
        "jit_specialization": "mock_tensor_exact_shape_stride_and_alignment",
        "layers": 48,
        "live_state_columns": 3,
        "num_warps": 4,
        "path_capacity": 16,
        "physical_rows": 32,
        "source_rows_per_request": 36,
        "state_columns": 34,
        "target": "sm_121a",
        "zero_tail": True,
    }
    incumbent = summary["variants"]["incumbent_metadata_copy"]["builds"]
    candidate = summary["variants"]["candidate_direct_input"]["builds"]
    expected = {
        "b1": {
            "registers": 34,
            "encoded": (744, 728),
            "static": (458, 445),
            "roundtrip": 17,
        },
        "b4": {
            "registers": 36,
            "encoded": (776, 744),
            "static": (488, 459),
            "roundtrip": 68,
        },
    }
    for batch, wanted in expected.items():
        before = incumbent[batch]
        after = candidate[batch]
        assert before["registers"] == after["registers"] == wanted["registers"]
        assert (before["encoded_sass_instructions"], after["encoded_sass_instructions"]) == wanted["encoded"]
        assert (before["static_sass_instructions"], after["static_sass_instructions"]) == wanted["static"]
        assert before["ldg"] == 12 and after["ldg"] == 11
        assert before["stg"] == 274 and after["stg"] == 272
        assert before["logical_work"]["metadata_intermediate_roundtrip_elements_per_event"] == wanted["roundtrip"]
        assert after["logical_work"]["metadata_intermediate_roundtrip_elements_per_event"] == 0
        for build in (before, after):
            assert build["stack_bytes"] == 0
            assert build["local_bytes"] == 0
            assert build["ldl"] == 0
            assert build["stl"] == 0
            assert build["calls"] == 0
