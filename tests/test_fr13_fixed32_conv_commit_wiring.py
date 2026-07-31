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


def _observed_runtime_tree(patcher_tree: ast.Module) -> ast.Module:
    sources = [
        node.value.value
        for node in patcher_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(sources) == 1
    return ast.parse(sources[0])


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


def test_fixed_route_uses_direct_path_and_generic_routes_stay_generic() -> None:
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


def test_fixed_route_delegates_row_guard_to_launcher_before_replay() -> None:
    tree = _fixed_route_tree(ast.parse(PATCHER_PATH.read_text()))
    route = _function(tree, "_fr13_fixed32_device_commit_route")
    calls = [
        _called_name(node) for node in ast.walk(route) if isinstance(node, ast.Call)
    ]

    assert "_fixed_conv_commit_rows" not in calls
    assert "validate_fixed32_conv_commit_rows" not in ast.unparse(route)
    assert calls.count("_fixed_conv_commit") == 1
    assert calls.count("_fixed_replay") == 1
    assert calls.index("_fixed_conv_commit") < calls.index("_fixed_replay")


def test_launcher_is_exactly_one_direct_kernel_without_host_sync() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    launcher = _function(tree, "launch_fixed32_conv_commit_to_col0")
    calls = [
        _called_name(node) for node in ast.walk(launcher) if isinstance(node, ast.Call)
    ]
    kernel_calls = [
        name
        for name in calls
        if name == "_fr13_fixed32_conv_direct_col0_kernel"
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

    assert kernel_calls == ["_fr13_fixed32_conv_direct_col0_kernel"]
    assert calls.index("validate_fixed32_conv_commit_rows") < calls.index(
        kernel_calls[0]
    )
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


def test_direct_commit_uses_logical_inner_strides_and_pregather_stays_logical() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    pregather = _function(tree, "_fr13_conv_col0_pregather_kernel")
    direct = _function(tree, "_fr13_fixed32_conv_direct_col0_kernel")
    pregather_source = ast.unparse(pregather)
    direct_source = ast.unparse(direct)

    assert "c_idx = offs // CONV_L" in pregather_source
    assert "l_idx = offs % CONV_L" in pregather_source
    assert "c_idx * s1 + l_idx * s2" in pregather_source
    assert "offs_c * bank_c_stride" in direct_source
    assert "state_col * bank_l_stride" in direct_source
    assert "(source_batch + source_row) * source_row_stride" in direct_source
    assert "for state_col in tl.static_range(0, CONV_L)" in direct_source


def test_direct_kernel_calls_exist_only_in_preseed_and_launcher() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    direct_name = "_fr13_fixed32_conv_direct_col0_kernel"
    owners: dict[str, list[str]] = {}

    for function in (node for node in tree.body if isinstance(node, ast.FunctionDef)):
        calls = [
            _called_name(node)
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and _called_name(node) == direct_name
        ]
        if calls:
            owners[function.name] = calls

    assert owners == {
        "preseed_fixed32_conv_col0_pregather": [direct_name],
        "launch_fixed32_conv_commit_to_col0": [direct_name],
    }


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
        "ssm_banks",
        "commit_spec_state_indices",
        "accepted_paths",
        "accepted_lens",
        "commit_source_stagings",
        "commit_state_src",
        "source_rows_per_batch",
    } <= argument_names
    assert {
        "commit_spec_state_indices",
        "accepted_paths",
        "accepted_lens",
    } <= identity_names
    assert "existing['ssm_banks']" in source
    assert "ssm_bank_refs" in source
    assert "zip(existing['ssm_banks'], ssm_bank_refs, strict=True)" in source
    assert "_fr13_fixed32_conv_direct_col0_kernel" in source
    assert "source_offsets" in source
    assert "direct_state_src" in source
    assert "commit_bank_overlap_policy" in source
    assert "commit_bank_partial_overlap" in source
    assert "commit_bank_alias_groups" in source
    assert "commit_bank_alias_width" in source
    assert "commit_bank_destination_guard" in source
    assert "commit_null_row_rejected" in source
    assert "commit_bank_nonoverlap" not in source


def test_patcher_runtime_identity_tuple_matches_kernel_state_order() -> None:
    tree = _observed_runtime_tree(ast.parse(PATCHER_PATH.read_text()))
    runtime = _function(tree, "_fr13_fixed32_conv_runtime_contract")
    assignments = {
        target.id: node.value
        for node in ast.walk(runtime)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"source_identity", "source_data_ptrs"}
    }
    assert set(assignments) == {"source_identity", "source_data_ptrs"}
    assert all(isinstance(value, ast.Tuple) for value in assignments.values())

    identity = assignments["source_identity"]
    assert isinstance(identity, ast.Tuple)
    assert [ast.unparse(value) for value in identity.elts] == [
        "tuple((id(bank) for bank in banks))",
        "tuple((id(bank) for bank in ssm_banks))",
        "id(offsets)",
        "id(bank_alias_ids_device)",
        "tuple((id(source) for source in sources))",
        "id(ssi_ptrs)",
        "id(ssi_strides)",
        "id(commit_spec_state_indices)",
        "id(accepted_paths)",
        "id(accepted_lens)",
        "tuple((id(source) for source in direct_sources))",
        "id(source_offsets)",
        "id(direct_state_src)",
        "id(staging)",
    ]
    data_ptrs = assignments["source_data_ptrs"]
    assert isinstance(data_ptrs, ast.Tuple)
    assert [ast.unparse(value) for value in data_ptrs.elts] == [
        "tuple((int(bank.data_ptr()) for bank in banks))",
        "tuple((int(bank.data_ptr()) for bank in ssm_banks))",
        ("tuple((int(bank.untyped_storage().data_ptr()) for bank in ssm_banks))"),
        "int(offsets.data_ptr())",
        "int(bank_alias_ids_device.data_ptr())",
        "tuple((int(source.data_ptr()) for source in sources))",
        "int(ssi_ptrs.data_ptr())",
        "int(ssi_strides.data_ptr())",
        "int(commit_spec_state_indices.data_ptr())",
        "int(accepted_paths.data_ptr())",
        "int(accepted_lens.data_ptr())",
        "tuple((int(source.data_ptr()) for source in direct_sources))",
        "int(source_offsets.data_ptr())",
        "int(direct_state_src.data_ptr())",
        "int(staging.data_ptr())",
    ]


def test_patcher_preseed_binds_companion_ssm_banks() -> None:
    tree = ast.parse(PATCHER_PATH.read_text())
    fragments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "_fr13_f32_pregather(" in node.value
    ]

    assert len(fragments) == 1
    assert "conv_banks=_fr13_f32_conv_banks" in fragments[0]
    assert "ssm_banks=_fr13_f32_banks" in fragments[0]
    assert "commit_source_stagings=" in fragments[0]
    assert "commit_state_src=" in fragments[0]
    assert "source_rows_per_batch=" in fragments[0]
