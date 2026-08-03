from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_gdn_prescaled_path_base_sm121a_codegen_20260803"
)
REVISION = "8959f328ce6b5e36c5eb6bbb1cb53c3c6e5f5bbe"


def test_artifact_is_reduced_offline_sm121a_evidence() -> None:
    files = {path.name for path in ARTIFACT.iterdir() if path.is_file()}
    assert {
        "README.md",
        "SHA256SUMS",
        "codegen_summary.json",
        "offline_codegen_audit.py",
        "opcode_delta.tsv",
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
    assert 'options={"num_warps": 8}' in compiler
    assert '"PRESCALED_PATH_BASE": VARIANTS[label]' in compiler
    assert f'EXPECTED_REVISION = "{REVISION}"' in compiler
    assert '"q": "*bf16"' in compiler
    assert "cuobjdump" in verifier
    assert "fresh-cache summary differs" in verifier


def test_summary_retains_resources_and_reduces_address_codegen() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["revision"] == REVISION
    assert summary["source_sha256"] == (
        "3fc393b93cd99299c93528481448fc4344f549fd958ed32b4e3fa85892b2f625"
    )
    assert summary["compile_contract"] == {
        "batches": [1, 4],
        "block_v": 8,
        "dim_k": 128,
        "dim_v": 128,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "flags_export": True,
        "groups": 5,
        "max_group_paths": 3,
        "max_path_len": 7,
        "num_key_heads": 4,
        "num_value_heads": 12,
        "num_warps": 8,
        "ordered_dynamic_loops": True,
        "physical_nodes": 32,
        "ring_export": True,
        "root_steps": 5,
        "signature": "explicit_deployed_pointer_types",
        "target": "sm_121a",
    }
    incumbent = summary["variants"]["incumbent_index_scaled"]["builds"]
    candidate = summary["variants"]["candidate_prescaled_path_base"][
        "builds"
    ]
    expected = {
        "b1": {
            "incumbent_cubin": (
                "f2a0925b92a04ecf3461b051180dcba728b4ca6b626d205ffc843fa04a0a4acb"
            ),
            "candidate_cubin": (
                "c393e02d9b34f5e6c7a60e944fe688f6cf2b3681c379502f001f903c47fb2e1d"
            ),
            "scale_operations": 2112,
        },
        "b4": {
            "incumbent_cubin": (
                "3104f923cacdfaae898c6d9c0ff2ca02bc6881c20ff8138547c014662daebc9c"
            ),
            "candidate_cubin": (
                "82068aed829c543b2c9db6136a8093b8550068227a2e2a1941e6457f62b77530"
            ),
            "scale_operations": 8448,
        },
    }
    for batch, wanted in expected.items():
        before = incumbent[batch]
        after = candidate[batch]
        assert before["cubin_sha256"] == wanted["incumbent_cubin"]
        assert after["cubin_sha256"] == wanted["candidate_cubin"]
        assert before["registers"] == after["registers"] == 99
        assert (before["sass_addressed_lines"], after["sass_addressed_lines"]) == (
            3552,
            3520,
        )
        assert (
            before["static_sass_instructions"],
            after["static_sass_instructions"],
        ) == (1776, 1760)
        assert (before["cubin_bytes"], after["cubin_bytes"]) == (
            136864,
            136560,
        )
        assert before["ldg"] == after["ldg"] == 62
        assert before["logical_work"][
            "path_base_scale_operations_per_event"
        ] == wanted["scale_operations"]
        assert after["logical_work"][
            "path_base_scale_operations_per_event"
        ] == 0
        assert before["node_source_sha256"] == after["node_source_sha256"]
        assert before["recurrence_source_sha256"] == after[
            "recurrence_source_sha256"
        ]
        for build in (before, after):
            assert build["stack_bytes"] == 0
            assert build["local_bytes"] == 0
            assert build["ldl"] == 0
            assert build["stl"] == 0
            assert build["calls"] == 0


def test_candidate_opcode_delta_is_address_favorable() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    before = summary["variants"]["incumbent_index_scaled"]["builds"]["b4"]
    after = summary["variants"]["candidate_prescaled_path_base"]["builds"][
        "b4"
    ]
    assert after["opcodes"]["IADD"] == before["opcodes"]["IADD"] - 6
    assert after["opcodes"]["IMAD"] == before["opcodes"]["IMAD"] - 3
    assert after["opcodes"]["LEA"] == before["opcodes"]["LEA"] - 3
    assert after["opcodes"]["SHF"] == before["opcodes"]["SHF"] - 3
