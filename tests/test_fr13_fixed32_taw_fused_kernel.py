from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_fused_kernel",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
taw = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(taw)


def _function(name: str) -> ast.FunctionDef:
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(definitions) == 1
    return definitions[0]


def _launched_kernel_name(call: ast.Call) -> str | None:
    if not isinstance(call.func, ast.Subscript):
        return None
    value = call.func.value
    return value.id if isinstance(value, ast.Name) else None


def test_fused_taw_has_fixed_two_kernel_launches_per_level() -> None:
    execute = _function("_fr13_fixed32_taw_execute_fused")
    loops = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert len(loops) == 1
    launches = [
        name
        for node in ast.walk(loops[0])
        if isinstance(node, ast.Call)
        if (name := _launched_kernel_name(node)) is not None
    ]
    assert launches == [
        "_fr13_fixed32_taw_chunk_stats_kernel",
        "_fr13_fixed32_taw_reduce_sample_commit_kernel",
    ]
    source = inspect.getsource(taw._fr13_fixed32_taw_execute_fused)
    for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize("):
        assert forbidden not in source

    census = taw._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
    assert census["walk_levels"] == 12
    assert census["chunk_stats_launches"] == 12
    assert census["reduce_sample_commit_launches"] == 12
    assert census["chunk_stats_programs_per_request"] == 12 * 2 * 64
    assert census["reduce_sample_commit_programs_per_request"] == 12
    assert census["full_vocab_probability_materializations"] == 0
    assert census["full_vocab_residual_materializations"] == 0
    assert census["host_syncs"] == 0


def test_fused_taw_dispatch_and_source_contract_pin_kernel_bodies() -> None:
    dispatcher = ast.unparse(_function("_fr13_fixed32_taw_execute"))
    assert "target_logits.is_cuda" in dispatcher
    assert "_fr13_fixed32_taw_execute_fused" in dispatcher
    assert "_fr13_fixed32_taw_execute_torch" in dispatcher
    assert taw._FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS == (
        "_fr13_fixed32_taw_chunk_stats_kernel",
        "_fr13_fixed32_taw_reduce_sample_commit_kernel",
    )

    contract = _function("_fr13_fixed32_taw_source_contract")
    canonical_keys = {
        key.value
        for node in ast.walk(contract)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert "kernels" in canonical_keys


def test_level_zero_fused_state_uses_a_boolean_alive_mask() -> None:
    kernel = _function("_fr13_fixed32_taw_reduce_sample_commit_kernel")
    level_zero = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "LEVEL"
        and isinstance(node.test.ops[0], ast.Eq)
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == 0
    )
    alive_assignment = next(
        node
        for node in level_zero.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "alive"
            for target in node.targets
        )
    )
    assert ast.unparse(alive_assignment.value) == "request == request"


def test_chunk_stats_zeroes_chunks_without_finite_logit_mass() -> None:
    kernel = ast.unparse(_function("_fr13_fixed32_taw_chunk_stats_kernel"))
    assert "chunk_has_finite_mass" in kernel
    assert "row_max > float('-inf')" in kernel
    assert "row_max < float('inf')" in kernel
    assert "valid & chunk_has_finite_mass" in kernel
    assert (
        "where(chunk_has_finite_mass, row_max, float('-inf'))"
        in kernel
    )


def _dense_and_sparse_residual(
    probabilities: np.ndarray,
    candidate_tokens: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlaps = probabilities[candidate_tokens]
    overlap_mass = float(overlaps.sum())
    if overlap_mass <= 0.0:
        return probabilities.copy(), probabilities.copy()

    q_mix = np.zeros_like(probabilities)
    np.add.at(q_mix, candidate_tokens, overlaps / overlap_mass)
    dense = np.maximum(probabilities - q_mix, 0.0)

    sparse = probabilities.copy()
    for token in np.unique(candidate_tokens):
        sparse[token] -= min(probabilities[token], q_mix[token])
    sparse = np.maximum(sparse, 0.0)
    if float(dense.sum()) <= 0.0:
        dense = probabilities.copy()
        sparse = probabilities.copy()
    return dense, sparse


@pytest.mark.parametrize(
    "candidate_tokens",
    (
        np.array([3, 17, 91]),
        np.array([5, 5, 5]),
        np.array([2, 2, 113]),
    ),
)
def test_sparse_duplicate_correction_matches_dense_qmix(
    candidate_tokens: np.ndarray,
) -> None:
    rng = np.random.default_rng(20260731)
    for _ in range(64):
        probabilities = rng.random(257)
        probabilities /= probabilities.sum()
        dense, sparse = _dense_and_sparse_residual(
            probabilities,
            candidate_tokens,
        )
        np.testing.assert_allclose(sparse, dense, rtol=0.0, atol=2e-17)


def test_sparse_residual_zero_and_zero_overlap_fallbacks() -> None:
    residual_zero = np.zeros(17)
    residual_zero[[2, 5, 11]] = [0.2, 0.3, 0.5]
    dense, sparse = _dense_and_sparse_residual(
        residual_zero,
        np.array([2, 5, 11]),
    )
    np.testing.assert_array_equal(dense, residual_zero)
    np.testing.assert_array_equal(sparse, residual_zero)

    zero_overlap = np.zeros(17)
    zero_overlap[13] = 1.0
    dense, sparse = _dense_and_sparse_residual(
        zero_overlap,
        np.array([2, 5, 11]),
    )
    np.testing.assert_array_equal(dense, zero_overlap)
    np.testing.assert_array_equal(sparse, zero_overlap)


@pytest.mark.parametrize(
    "case",
    (
        "_fr13_fixed32_test_accept_leaf_depth_pad",
        "_fr13_fixed32_test_reject_residual_zero_mass",
        "_fr13_fixed32_test_duplicate_semantics",
    ),
)
def test_cpu_oracle_edge_cases_remain_exact(case: str) -> None:
    topology = taw._fr13_fixed32_topology()
    mode = "tail6_fixed32"
    taw.fr13_fixed32_taw_preseed(
        "cpu",
        mode=mode,
        valid_mask=int(topology.VALID_MASK_BY_MODE[mode]),
    )
    taw.fr13_fixed32_taw_set_work_callback(lambda _payload: None)
    try:
        getattr(taw, case)(topology)
    finally:
        taw.fr13_fixed32_taw_set_work_callback(None)
