from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_cfwd_ownerpath_warp32_sm121a_codegen_20260803"
)
SUMMARY = ARTIFACT / "codegen_summary.json"
KERNEL = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"


def _function_source(name: str) -> str:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(source, node) or ""


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
    assert "create_function_from_signature" in compiler
    assert "nvdisasm" in verifier
    assert "fresh-cache summary differs" in verifier


def test_summary_binds_incumbent_defect_and_warp32_fix() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="ascii"))
    assert summary["compile_contract"] == {
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
    variants = summary["variants"]
    expected = {
        "incumbent": {
            "b1": (26, 0, 144, 10, 1, 0, 0, 0),
            "b4": (26, 0, 176, 10, 1, 0, 0, 0),
        },
        "superseded_v3": {
            "b1": (18, 8, 152, 7, 1, 2, 2, 2),
            "b4": (17, 8, 184, 7, 1, 2, 2, 2),
        },
        "candidate": {
            "b1": (18, 0, 144, 8, 1, 0, 0, 0),
            "b4": (16, 0, 168, 8, 1, 0, 0, 0),
        },
    }
    for label, builds in expected.items():
        for batch, wanted in builds.items():
            build = variants[label]["builds"][batch]
            got = (
                build["registers"],
                build["launch_shared_bytes"],
                build["encoded_sass_instructions"],
                build["ldg"],
                build["stg"],
                build["bar"],
                build["lds"],
                build["sts"],
            )
            assert got == wanted
            assert build["stack_bytes"] == 0
            assert build["local_bytes"] == 0
            assert build["ldl"] == 0
            assert build["stl"] == 0
            assert build["calls"] == 0


def test_source_and_census_bind_warp_local_fail_closed_route() -> None:
    kernel = _function_source("_fr13_fixed32_conv_commit_row_guard_kernel")
    assert "alias_offsets = tl.arange(0, 32)" in kernel
    assert "aliases_lo_ok" in kernel
    assert "aliases_hi_ok" in kernel
    assert "ALIAS_CAP" not in kernel
    source = KERNEL.read_text(encoding="utf-8")
    census = (ROOT / "scripts/fr13_fixed32_work_census.py").read_text()
    route = "fixed32_triton_alias3_ownerpath_warp32_physical32_v4"
    assert route in source
    assert route in census
    assert "CONV_ROW_GUARD_ALIAS_VECTOR_LOADS_PER_EVENT = 2" in census


def test_fixed32_b1_b4_work_scaling_is_explicit() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="ascii"))
    incumbent = summary["variants"]["incumbent"]["builds"]
    candidate = summary["variants"]["candidate"]["builds"]
    assert incumbent["b1"]["logical_work"][
        "source_visible_values_before_compiler_cache_effects"
    ] == 2976
    assert incumbent["b4"]["logical_work"][
        "source_visible_values_before_compiler_cache_effects"
    ] == 17088
    assert candidate["b1"]["logical_work"][
        "source_visible_values_before_compiler_cache_effects"
    ] == 1937
    assert candidate["b4"]["logical_work"][
        "source_visible_values_before_compiler_cache_effects"
    ] == 11060
    assert candidate["b1"]["logical_work"]["kernel_launches_per_event"] == 1
    assert candidate["b4"]["logical_work"]["kernel_launches_per_event"] == 1
    assert candidate["b1"]["logical_work"]["alias_id_values"] == 48
    assert candidate["b4"]["logical_work"]["alias_id_values"] == 48
