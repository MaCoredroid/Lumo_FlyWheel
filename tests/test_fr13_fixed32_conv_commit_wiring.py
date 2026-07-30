from __future__ import annotations

import ast
from pathlib import Path

KERNEL_PATH = Path("src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py")
PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


def _fixed_route_tree(patcher_tree: ast.Module) -> ast.Module:
    sources = [
        node
        for node in ast.walk(patcher_tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "def _fr13_fixed32_device_commit_route" in node.value
    ]
    assert len(sources) == 1
    return ast.parse(sources[0].value)


def _called_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return ast.unparse(call.func)
    if isinstance(call.func, ast.Subscript):
        return ast.unparse(call.func.value)
    return ast.unparse(call.func)


def _direct_call_name(statement: ast.stmt) -> str | None:
    if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        return _called_name(statement.value)
    return None


def test_fixed_route_uses_two_launch_path_and_generic_routes_stay_generic() -> None:
    tree = _fixed_route_tree(ast.parse(PATCHER_PATH.read_text()))
    route = _function(tree, "_fr13_fixed32_device_commit_route")
    route_calls = [
        _called_name(node) for node in ast.walk(route) if isinstance(node, ast.Call)
    ]
    all_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and _called_name(node) == "_fr13_conv_commit_to_col0"
    ]

    assert "_fixed_conv_commit" in route_calls
    assert "_fr13_conv_commit_to_col0" not in route_calls
    assert len(all_calls) == 3


def test_fixed_route_guards_rows_before_raw_conv_commit_and_replay() -> None:
    tree = _fixed_route_tree(ast.parse(PATCHER_PATH.read_text()))
    route = _function(tree, "_fr13_fixed32_device_commit_route")
    calls = [
        _called_name(node) for node in ast.walk(route) if isinstance(node, ast.Call)
    ]

    assert calls.count("_fixed_conv_commit_rows") == 1
    assert calls.count("_fixed_conv_commit") == 1
    assert calls.count("_fixed_replay") == 1
    assert calls.index("_fixed_conv_commit_rows") < calls.index("_fixed_conv_commit")
    assert calls.index("_fixed_conv_commit") < calls.index("_fixed_replay")


def test_launcher_is_exactly_gather_then_scatter_without_host_sync() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    launcher = _function(tree, "launch_fixed32_conv_commit_to_col0")
    calls = [
        _called_name(node) for node in ast.walk(launcher) if isinstance(node, ast.Call)
    ]
    kernel_calls = [
        name for name in calls if name.startswith("_fr13_fixed32_conv_commit_")
    ]
    banned_suffixes = (
        ".item",
        ".tolist",
        ".cpu",
        ".numpy",
        ".synchronize",
        ".nonzero",
    )
    source = ast.unparse(launcher)

    assert kernel_calls == [
        "_fr13_fixed32_conv_commit_gather_kernel",
        "_fr13_fixed32_conv_commit_scatter_kernel",
    ]
    assert not any(name.endswith(banned_suffixes) for name in calls), calls
    assert not any(isinstance(node, ast.Try) for node in ast.walk(launcher))
    assert "for bank in conv_banks" not in source
    assert "for bank in state['banks']" not in source
    assert source.count(".data_ptr()") == 3


def test_precommit_row_guard_has_no_host_sync() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    guard = _function(tree, "validate_fixed32_conv_commit_rows")
    calls = [
        _called_name(node) for node in ast.walk(guard) if isinstance(node, ast.Call)
    ]
    banned_suffixes = (
        ".item",
        ".tolist",
        ".cpu",
        ".numpy",
        ".synchronize",
        ".nonzero",
    )

    assert not any(name.endswith(banned_suffixes) for name in calls), calls
    assert "_fr13_fixed32_device_assert" in calls


def test_commit_kernels_are_physical_and_pregather_remains_logical() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    pregather = _function(tree, "_fr13_conv_col0_pregather_kernel")
    gather = _function(tree, "_fr13_fixed32_conv_commit_gather_kernel")
    scatter = _function(tree, "_fr13_fixed32_conv_commit_scatter_kernel")
    pregather_source = ast.unparse(pregather)

    assert "c_idx = offs // CONV_L" in pregather_source
    assert "l_idx = offs % CONV_L" in pregather_source
    assert "c_idx * s1 + l_idx * s2" in pregather_source
    for commit, row_name in ((gather, "src_row"), (scatter, "dst_row")):
        argument_names = {argument.arg for argument in commit.args.args}
        names = {node.id for node in ast.walk(commit) if isinstance(node, ast.Name)}
        source = ast.unparse(commit)

        assert {"s1", "s2", "CONV_L"}.isdisjoint(argument_names)
        assert {"c_idx", "l_idx", "s1", "s2", "CONV_L"}.isdisjoint(names)
        assert f"{row_name}.to(tl.int64) * row_stride + offs" in source


def test_commit_kernel_calls_exist_only_as_adjacent_preseed_and_launcher_pairs() -> (
    None
):
    tree = ast.parse(KERNEL_PATH.read_text())
    gather_name = "_fr13_fixed32_conv_commit_gather_kernel"
    scatter_name = "_fr13_fixed32_conv_commit_scatter_kernel"
    expected_pair = [gather_name, scatter_name]
    owners: dict[str, list[str]] = {}

    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        calls = [
            _called_name(node)
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _called_name(node) in expected_pair
        ]
        if calls:
            owners[function.name] = calls

    assert owners == {
        "preseed_fixed32_conv_col0_pregather": expected_pair,
        "launch_fixed32_conv_commit_to_col0": expected_pair,
    }
    for owner_name in owners:
        owner = _function(tree, owner_name)
        adjacent_pairs = 0
        for node in ast.walk(owner):
            body = getattr(node, "body", None)
            if not isinstance(body, list):
                continue
            direct_names = [_direct_call_name(statement) for statement in body]
            adjacent_pairs += sum(
                direct_names[index : index + 2] == expected_pair
                for index in range(len(direct_names) - 1)
            )
        assert adjacent_pairs == 1, owner_name

        for call in (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call) and _called_name(node) in expected_pair
        ):
            assert all(keyword.arg != "CONV_L" for keyword in call.keywords)


def test_full_48_bank_lease_audit_is_outside_the_event_launcher() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    audit = _function(tree, "audit_fixed32_conv_commit_lease")
    launcher = _function(tree, "launch_fixed32_conv_commit_to_col0")

    assert "_validate_fixed32_conv_pregather_preseed" in ast.unparse(audit)
    assert "_validate_fixed32_conv_pregather_preseed" not in ast.unparse(launcher)


def test_preseed_binds_exact_commit_operands_and_warms_b1_through_b4() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    preseed = _function(tree, "preseed_fixed32_conv_col0_pregather")
    argument_names = {argument.arg for argument in preseed.args.kwonlyargs}
    source = ast.unparse(preseed)
    identity_names = {
        node.comparators[0].id
        for node in ast.walk(preseed)
        if isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.IsNot)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Name)
    }

    assert {
        "commit_spec_state_indices",
        "accepted_paths",
        "accepted_lens",
    } <= argument_names
    assert {
        "commit_spec_state_indices",
        "accepted_paths",
        "accepted_lens",
    } <= identity_names
    assert "_fr13_fixed32_conv_commit_gather_kernel" in source
    assert "_fr13_fixed32_conv_commit_scatter_kernel" in source
    assert "commit_bank_nonoverlap" in source
