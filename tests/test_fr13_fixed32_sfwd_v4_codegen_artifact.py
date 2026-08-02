from __future__ import annotations

import ast
from pathlib import Path


ARTIFACT = Path(
    "results/fr13_fixed32_sfwd_v4_sm121a_codegen_20260802"
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


def test_offline_codegen_binds_exact_deployment_geometry_without_gpu_calls() -> None:
    source = (ARTIFACT / "offline_codegen_audit.py").read_text()
    tree = ast.parse(source)

    assert _literal_assignment(tree, "DEPLOYMENT_CONFIGS") == {
        1: {"block_c": 128, "num_warps": 2},
        4: {"block_c": 256, "num_warps": 4},
    }
    constants = _literal_assignment(tree, "BASE_CONSTANTS")
    assert constants == {
        "CONV_STRIDE_ROW": 348160,
        "N": 32,
        "C": 10240,
        "WIDTH": 4,
        "STATE_LEN": 34,
        "SOURCE_ROWS": 36,
        "HAS_BIAS": False,
        "X_STRIDE_ROW": 16384,
    }
    target_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "GPUTarget"
    ]
    assert len(target_calls) == 1
    assert tuple(ast.literal_eval(arg) for arg in target_calls[0].args) == (
        "cuda",
        121,
        32,
    )
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "torch.cuda" not in source


def test_offline_verifier_enforces_sm121a_and_spill_free_codegen() -> None:
    source = (ARTIFACT / "verify_codegen_outputs.py").read_text()

    assert "(?m)^\\.target sm_121a$" in source
    assert '"stack_bytes"]' in source
    assert '"local_bytes"]' in source
    assert '"ldl"]' in source
    assert '"stl"]' in source
    assert '"calls"]' in source
    assert "primary/rebuild verification report" in source
