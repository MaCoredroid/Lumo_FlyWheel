from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path


MODULE_PATH = Path("scripts/fr13_device_multidraft_kernel.py")
SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_taw_exact_commit_kernel",
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


def _assignment_values(function: ast.FunctionDef, name: str) -> list[str]:
    values = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            values.append(ast.dump(node.value, include_attributes=False))
    return values


def _method_calls(function: ast.FunctionDef, name: str) -> list[str]:
    calls = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == name
        ):
            calls.append(ast.dump(call, include_attributes=False))
    return calls


def test_exact_cuda_reuses_every_floating_sampling_expression() -> None:
    oracle = _function("_fr13_fixed32_taw_execute_torch")
    candidate = _function("_fr13_fixed32_taw_execute_exact_cuda")
    numerical_assignments = (
        "self_indices",
        "self_prob",
        "self_token",
        "first_child",
        "target_indices",
        "target_prob",
        "kid_tokens",
        "kid_mask",
        "overlaps",
        "overlap_mass",
        "zero_mass",
        "source",
        "selected_token",
        "same_token",
        "q_mix_token",
        "target_at_token",
        "accept_probability",
        "accepted",
        "weights",
        "q_mix_vocab",
        "residual",
        "residual_mass",
        "rejected_token",
    )
    for name in numerical_assignments:
        oracle_values = _assignment_values(oracle, name)
        candidate_values = _assignment_values(candidate, name)
        assert oracle_values
        assert candidate_values == oracle_values, name
    assert _method_calls(candidate, "q_mix_vocab") == _method_calls(
        oracle,
        "q_mix_vocab",
    )


def test_exact_commit_kernel_contains_no_floating_sampling_math() -> None:
    kernel = _function("_fr13_fixed32_taw_exact_commit_kernel")
    source = ast.unparse(kernel)
    for forbidden in (
        "tl.exp",
        "tl.exp2",
        "tl.log",
        "tl.sum(",
        "tl.max(",
        "tl.cumsum(",
        "float32",
        "1e-30",
    ):
        assert forbidden not in source
    for required in (
        "output_tokens",
        "output_lens",
        "accepted_path_rows",
        "accepted_lens",
        "last_row",
        "current_state",
        "alive_state",
    ):
        assert required in source


def test_exact_cuda_has_one_integer_commit_launch_per_level() -> None:
    execute = _function("_fr13_fixed32_taw_execute_exact_cuda")
    loops = [
        node
        for node in ast.walk(execute)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    assert len(loops) == 1
    walk_loop = next(
        loop
        for loop in loops
        if isinstance(loop.target, ast.Name) and loop.target.id == "level"
    )
    launches = [
        name
        for node in ast.walk(walk_loop)
        if isinstance(node, ast.Call)
        if (name := _launched_kernel_name(node)) is not None
    ]
    assert launches == ["_fr13_fixed32_taw_exact_commit_kernel"]
    source = inspect.getsource(taw._fr13_fixed32_taw_execute_exact_cuda)
    for forbidden in (".item(", ".cpu(", ".tolist(", "synchronize("):
        assert forbidden not in source


def test_exact_commit_dispatch_and_census_are_fail_closed() -> None:
    dispatcher = ast.unparse(_function("_fr13_fixed32_taw_execute"))
    assert "target_logits.is_cuda" in dispatcher
    assert "_fr13_fixed32_taw_execute_exact_cuda" in dispatcher
    assert "_fr13_fixed32_taw_execute_torch" in dispatcher
    assert taw._FR13_FIXED32_TAW_KERNEL_SOURCE_FUNCTIONS == (
        "_fr13_fixed32_taw_exact_commit_kernel",
    )
    census = taw._FR13_FIXED32_TAW_TENSOR_CALL_CENSUS
    assert census["walk_levels"] == 12
    assert census["full_vocab_softmax_calls"] == 24
    assert census["full_vocab_cdf_calls"] == 24
    assert census["source_cdf_calls"] == 12
    assert census["exact_commit_launches"] == 12
    assert census["exact_commit_programs_per_request"] == 12
    assert census["floating_sampling_reimplementation"] is False
    assert census["output_scatter_calls"] == 0
    assert census["path_scatter_calls"] == 0
