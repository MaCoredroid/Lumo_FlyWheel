from __future__ import annotations

import ast
import json
from pathlib import Path


ARTIFACT = Path(
    "results/fr13_fixed32_treeconv_zero_tail_sm121a_codegen_20260803"
)


def _literal_assignment(tree: ast.Module, name: str):
    node = next(
        statement
        for statement in tree.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == name
    )
    return ast.literal_eval(node.value)


def test_offline_codegen_binds_fixed32_k64_root1_sm121a_without_gpu_calls() -> None:
    source = (ARTIFACT / "offline_codegen_audit.py").read_text()
    tree = ast.parse(source)

    assert _literal_assignment(tree, "BASE_CONSTANTS") == {
        "CONV_C": 10240,
        "CONV_L": 34,
        "SOURCE_ROWS": 36,
        "ELEM_BYTES": 2,
        "SPEC_COLS": 32,
        "PATH_COLS": 16,
        "BLOCK_C": 1024,
    }
    assert _literal_assignment(tree, "DEPLOYMENT_CONFIGS") == {
        1: {"num_warps": 4},
        4: {"num_warps": 4},
    }
    assert _literal_assignment(tree, "DEPLOYMENT_CONTEXT") == {
        "fixed_physical_rows": 32,
        "drafter_vocab_k": 65536,
        "root_reduction": 1,
    }
    target_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GPUTarget"
    ]
    assert len(target_calls) == 2
    assert all(
        tuple(ast.literal_eval(arg) for arg in call.args) == ("cuda", 121, 32)
        for call in target_calls
    )
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != ""' in source
    assert "torch.cuda" not in source


def test_offline_verifier_enforces_target_spills_identity_and_traffic() -> None:
    source = (ARTIFACT / "verify_codegen_outputs.py").read_text()

    assert "(?m)^\\.target sm_121a$" in source
    assert 'expected["stack_bytes"]' in source
    assert 'expected["local_bytes"]' in source
    assert 'expected["ldl"]' in source
    assert 'expected["stl"]' in source
    assert 'expected["calls"]' in source
    assert "selector-off machine-code identity" in source
    assert "source_read_bytes_per_event" in source
    assert "primary/rebuild report identity" in source


def test_checked_summary_records_reproducible_b1_b4_codegen() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())

    assert summary["verified"] is True
    assert summary["offline_only"] is True
    assert summary["timing_claim"] is False
    assert summary["deployment_context"] == {
        "fixed_physical_rows": 32,
        "drafter_vocab_k": 65536,
        "root_reduction": 1,
    }
    assert summary["fixed32_route"] == {
        "candidate_default_off": True,
        "direct_committed_rows": {"b1": 48, "b4": 192},
        "full_node_writebacks_per_event": 0,
        "generic_batched_writeback_guarded_out": True,
    }

    for kind, incumbent_ldg, candidate_ldg, stores in (
        ("direct", 311, 32, 272),
        ("metadata", 312, 33, 274),
    ):
        for batch_key, batch in (("b1", 1), ("b4", 4)):
            variants = summary["absolute_codegen"][kind]
            incumbent = variants["incumbent"][batch_key]
            retained = variants["retained_off"][batch_key]
            candidate = variants["candidate"][batch_key]
            assert candidate["ctas_per_event"] == 480 * batch
            assert incumbent["source_columns_loaded_per_row"] == 34
            assert candidate["source_columns_loaded_per_row"] == 3
            assert candidate["destination_columns_stored_per_row"] == 34
            assert incumbent["ldg"] == incumbent_ldg
            assert candidate["ldg"] == candidate_ldg
            assert incumbent["stg"] == candidate["stg"] == stores
            assert retained["sass_sha256"] == incumbent["sass_sha256"]
            assert candidate["stack_bytes"] == candidate["local_bytes"] == 0
            assert candidate["ldl"] == candidate["stl"] == candidate["calls"] == 0
            comparison = summary["comparisons"][kind][batch_key]
            assert comparison["selector_off_machine_code_identity"] is True
            assert comparison["source_read_bytes_saved"] == 30_474_240 * batch

    generic = summary["retained_generic"]
    assert generic["b1_ctas_per_event"] == 15360
    assert generic["b4_ctas_per_event"] == 61440
