from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)
PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"


def _tree_and_source(path: Path = KERNEL_PATH) -> tuple[ast.Module, str]:
    source = path.read_text(encoding="utf-8")
    return ast.parse(source), source


def _function_source(name: str, path: Path = KERNEL_PATH) -> str:
    tree, source = _tree_and_source(path)
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def _observed_runtime_function_source(name: str) -> str:
    tree, _source = _tree_and_source(PATCHER_PATH)
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "_FR13_FIXED32_OBSERVED_RUNTIME_SOURCE"
            for target in node.targets
        )
    )
    runtime_source = ast.literal_eval(assignment.value)
    runtime_tree = ast.parse(runtime_source)
    node = next(
        item
        for item in runtime_tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(runtime_source, node)
    assert segment is not None
    return segment


def _load_contract_namespace() -> dict[str, object]:
    constants = {
        "_FR13_FIXED32_PARENT",
        "_FR13_FIXED32_SUBTREE_LEVELS",
        "_FR13_FIXED32_EXPORT_NODES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID",
        "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS",
    }
    functions = {
        "_fr13_canonical_sha256",
        "_fr13_fixed32_gdn_single_launch_contract",
    }
    tree, _source = _tree_and_source()
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    namespace = {"hashlib": hashlib, "json": json}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )
    assert constants | functions <= namespace.keys()
    return namespace


def _load_selector_namespace() -> dict[str, object]:
    constants = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_SIDECARS",
    }
    function = "_fr13_resolve_fixed32_gdn_single_launch"
    tree, _source = _tree_and_source()
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name == function)
    ]
    namespace = {"os": os}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace


def test_depth_first_contract_covers_each_node_and_writer_once() -> None:
    namespace = _load_contract_namespace()
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]
    contract = namespace["_fr13_fixed32_gdn_single_launch_contract"](levels)

    assert contract["candidate"] == "fixed32_gdn_single_launch_tree_v1"
    assert contract["root_nodes"] == (0, 1, 4, 9, 14)
    assert contract["branch_path_indices"] == (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (0, 9, 10),
    )
    assert contract["group_sizes"] == (2, 2, 2, 2, 3)
    assert contract["launches"] == 1
    assert contract["physical_programs"] == 1
    assert contract["physical_grid_z"] == (1,)
    assert contract["node_updates"] == 32
    assert contract["critical_node_steps"] == 32
    assert contract["live_state_tiles"] == 2
    assert contract["nominal_register_fp32_values_per_cta"] == 4096
    assert contract["nominal_register_fp32_values_per_thread"] == 16
    assert contract["state_export_writes"] == 0
    assert contract["state_parent_reads"] == 0
    assert contract["single_writer_nodes"] == 32

    level1 = levels[1]
    execution = []
    for parent, indices in namespace[
        "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS"
    ]:
        execution.append(parent)
        for index in indices:
            assert level1[index][1] == parent
            execution.extend(level1[index][0])
    assert sorted(execution) == list(range(32))
    assert all(execution.count(node) == 1 for node in range(32))


def test_depth_first_contract_rejects_wrong_parent_and_duplicate_path() -> None:
    namespace = _load_contract_namespace()
    validate = namespace["_fr13_fixed32_gdn_single_launch_contract"]
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]

    with pytest.raises(RuntimeError, match="parent/path mismatch"):
        validate(
            levels,
            groups=(
                (0, (0, 2)),
                (1, (3, 4)),
                (4, (5, 6)),
                (9, (7, 8)),
                (14, (1, 9, 10)),
            ),
        )
    with pytest.raises(RuntimeError, match="branch coverage drift"):
        validate(
            levels,
            groups=(
                (0, (1, 1)),
                (1, (3, 4)),
                (4, (5, 6)),
                (9, (7, 8)),
                (14, (0, 9, 10)),
            ),
        )


def test_single_launch_selector_is_default_off_and_fail_closed(tmp_path) -> None:
    namespace = _load_selector_namespace()
    resolve = namespace["_fr13_resolve_fixed32_gdn_single_launch"]
    kwargs = {
        "fixed32_mode": "hydra27_fixed32",
        "sidecars": (),
        "geom_override": {"BV": 8},
    }

    assert resolve(environ={}, **kwargs) is False
    assert resolve(
        environ={"FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "1"}, **kwargs
    ) is True
    with pytest.raises(RuntimeError, match="must be exactly 1"):
        resolve(
            environ={"FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "0"}, **kwargs
        )
    with pytest.raises(RuntimeError, match="requires an exact fixed32"):
        resolve(
            None,
            environ={"FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "1"},
            sidecars=(),
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="pinned exactly"):
        resolve(
            environ={"FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "1"},
            sidecars=(),
            geom_override={"BV": 16},
            fixed32_mode="tail6_fixed32",
        )

    sidecar = tmp_path / "single-launch.arm"
    sidecar.write_text("1\n", encoding="ascii")
    assert resolve(
        environ={},
        sidecars=(sidecar,),
        fixed32_mode="tail6_fixed32",
        geom_override={"BV": 8},
    ) is True


def test_legacy_batched_selectors_fail_closed_for_single_launch(
    tmp_path: Path,
) -> None:
    constants = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED",
        "_FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT",
        "_FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB_ENABLED",
        "_FR13_FIXED32_BATCH_GDN_PRODUCTION_ARM",
    }
    functions = {
        "_fr13_resolve_fixed32_batch_gdn_bv",
        "_fr13_fixed32_batch_gdn_byte_ab_control",
        "_fr13_fixed32_batch_gdn_graph_byte_ab_control",
        "_fr13_fixed32_batch_gdn_production_control",
    }
    tree, _source = _tree_and_source()
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in node.targets
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    namespace = {
        "os": os,
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH": True,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )

    with pytest.raises(RuntimeError, match="legacy batched-GDN selector"):
        namespace["_fr13_resolve_fixed32_batch_gdn_bv"](
            "hydra27_fixed32",
            env_name="FR13_FIXED32_BATCH_GDN_BV_CANDIDATE",
            sidecars=(),
            environ={"FR13_FIXED32_BATCH_GDN_BV_CANDIDATE": "8"},
            geom_override={"BV": 8},
            allow_reference_bv=True,
        )

    enabled = tmp_path / "legacy-eager.enabled"
    enabled.write_text("1\n", encoding="ascii")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB_ENABLED_PATH", str(enabled)
        )
        monkeypatch.setenv(
            "FR13_FIXED32_BATCH_GDN_BYTE_AB_REAL_EVENT_PATH",
            str(tmp_path / "absent-event"),
        )
        with pytest.raises(RuntimeError, match="legacy diagnostic"):
            namespace["_fr13_fixed32_batch_gdn_byte_ab_control"]()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_GRAPH_BYTE_AB", "1")
        with pytest.raises(RuntimeError, match="legacy diagnostic"):
            namespace["_fr13_fixed32_batch_gdn_graph_byte_ab_control"]()

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("FR13_FIXED32_BATCH_GDN_PRODUCTION", "1")
        with pytest.raises(RuntimeError, match="legacy credential"):
            namespace["_fr13_fixed32_batch_gdn_production_control"]()

    selector = _function_source("fixed32_batch_gdn_selector")
    assert "if _FR13_FIXED32_GDN_SINGLE_LAUNCH:" in selector
    assert "_fr13_fixed32_batch_gdn_byte_ab_control()" in selector
    assert "_fr13_fixed32_batch_gdn_graph_byte_ab_control()" in selector
    assert "_fr13_fixed32_batch_gdn_production_control()" in selector


def test_kernel_interleaves_root_and_branch_with_two_state_tiles() -> None:
    helper = _function_source("_tree_gdn_fixed32_single_launch_node")
    kernel = _function_source("_tree_gdn_kernel_fixed32_single_launch")

    assert "_gdn_node_step(" in helper
    assert "tl.store(\n        out" in helper
    assert "state_export" not in helper
    assert "return tl.where(n_ok, new_state, state_i)" in helper
    assert helper.index("b_a_log = b_g") < helper.index("if RAW_GATING:")
    assert helper.index("b_dt_bias = b_b") < helper.index("if RAW_GATING:")
    assert "b_a_log = global_a_log" in helper
    assert "b_dt_bias = global_dt_bias" in helper
    assert "for root_index in tl.static_range(0, ROOT_STEPS):" in kernel
    assert kernel.index("root_state = _tree_gdn_fixed32_single_launch_node(") < (
        kernel.index("for member in tl.static_range")
    )
    assert "branch_state = root_state" in kernel
    assert "for path_offset in tl.range(0, path_len):" in kernel
    assert kernel.count("_tree_gdn_fixed32_single_launch_node(") == 2
    assert "state_export" not in kernel
    assert "tl.atomic_add(" in kernel
    assert "flag_writer = (pid_vh == 0) & (pid_v == 0) & (pid_batch == 0)" in kernel
    assert kernel.index("b_a_log = tl.load(") < kernel.index(
        "for root_index in tl.static_range"
    )


def test_b1_b4_launchers_use_one_grid_and_reference_forces_incumbent() -> None:
    b1 = _function_source("launch_tree_gdn_prepared")
    b4 = _function_source("launch_tree_gdn_prepared_fixed32_batch")

    assert "_tree_gdn_kernel_fixed32_single_launch[" in b1
    assert "triton.cdiv(dim_v, _path_block_v),\n                    1," in b1
    assert "COUNT_INVOCATION=_count" in b1
    assert "FLAGS_EXPORT=_flags_export" in b1
    assert "if force_reference_structure" in b1

    assert "_tree_gdn_kernel_fixed32_single_launch[" in b4
    assert "triton.cdiv(dim_v, _block_v),\n                    batch," in b4
    assert "COUNT_INVOCATION=count_invocation" in b4
    assert "FLAGS_EXPORT=flags_export" in b4
    assert "force_reference_structure=True" in b4
    assert b4.index("force_reference_structure=True") > b4.index(
        "def _launch_reference"
    )


@pytest.mark.parametrize("batch", [1, 4])
def test_single_launch_static_traffic_and_parallelism(batch: int) -> None:
    layers = 48
    value_heads = 48
    dim_v = 128
    dim_k = 128
    block_v = 8
    state_bytes = value_heads * dim_v * dim_k * 4
    ctas_per_unit = batch * layers * value_heads * (dim_v // block_v)

    reference_ctas = 12 * ctas_per_unit
    parent_group_ctas = 6 * ctas_per_unit
    single_launch_ctas = ctas_per_unit
    reference_handoff = batch * layers * (5 + 11) * state_bytes
    parent_group_handoff = batch * layers * (5 + 5) * state_bytes
    single_launch_handoff = 0

    assert state_bytes == 3_145_728
    nominal_register_fp32_values = 4_096
    assert nominal_register_fp32_values * 4 == 16_384
    assert nominal_register_fp32_values // (8 * 32) == 16
    assert single_launch_handoff == 0
    if batch == 1:
        assert (reference_ctas, parent_group_ctas, single_launch_ctas) == (
            442_368,
            221_184,
            36_864,
        )
        assert (reference_handoff, parent_group_handoff) == (
            2_415_919_104,
            1_509_949_440,
        )
    else:
        assert (reference_ctas, parent_group_ctas, single_launch_ctas) == (
            1_769_472,
            884_736,
            147_456,
        )
        assert (reference_handoff, parent_group_handoff) == (
            9_663_676_416,
            6_039_797_760,
        )


def test_observer_separates_logical_and_physical_critical_paths() -> None:
    observer = _observed_runtime_function_source(
        "_fr13_fixed32_observed_gdn"
    )
    validator = _observed_runtime_function_source(
        "_fr13_fixed32_validate_forward_work"
    )
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert 'physical_route = "fixed32_single_launch_tree"' in observer
    assert 'physical_critical_path = normalized_single_launch[' in observer
    assert 'event["gdn_critical_path"] = normalized_contract["critical"]' in observer
    assert 'event["gdn_physical_critical_path"] = physical_critical_path' in observer
    assert 'expected_physical_critical_path = 32' in validator
    assert '"gdn_critical_path": 12' in validator
    assert (
        '"gdn_physical_critical_path": expected_physical_critical_path'
        in validator
    )
    assert '"physical_recurrence_critical_path": int(' in patcher
    assert '"fixed32_single_launch_contract": (' in patcher


def test_observer_records_single_launch_physical_work() -> None:
    observer_source = _observed_runtime_function_source(
        "_fr13_fixed32_observed_gdn"
    )
    event = {
        "batch_size": 1,
        "gdn_calls": set(),
        "gdn_layers": set(),
        "gdn_scan_calls": 0,
        "gdn_launches": 0,
        "gdn_path_programs": 0,
        "gdn_padded_slots": 0,
        "gdn_physical_route": None,
        "gdn_physical_programs": 0,
        "gdn_physical_grid_z": None,
        "gdn_level1_parent_loads": 0,
        "gdn_single_writer_nodes": 0,
        "gdn_nodes": 0,
        "gdn_critical_path": None,
        "gdn_physical_critical_path": None,
        "gdn_grid_z": None,
        "gdn_max_path_lengths": None,
        "gdn_export_or_mask": None,
        "gdn_parent_sha256": None,
        "gdn_ancestry_sha256": None,
    }
    namespace = {
        "_fr13_fixed32_observed_work_target": (
            lambda _label, _capturing, _batch: (event, None)
        )
    }
    exec(compile(observer_source, "<observer>", "exec"), namespace)
    runtime_state = {
        "schedule": "fixed32",
        "route_armed": True,
        "n_levels": 2,
        "critical": 12,
        "parent_nodes": 32,
        "emask_rows": 32,
        "export_rows": 32,
        "fixed32_contract": {
            "path_counts": (1, 11),
            "max_lengths": (5, 7),
            "launches": 2,
            "programs": 12,
            "padded_slots": 82,
            "critical": 12,
            "export_or_mask": 16915,
            "parent_sha256": "a" * 64,
            "ancestry_sha256": "b" * 64,
        },
        "fixed32_parent_group_contract": None,
        "fixed32_single_launch_contract": {
            "candidate": "fixed32_gdn_single_launch_tree_v1",
            "root_nodes": (0, 1, 4, 9, 14),
            "branch_path_indices": (
                (1, 2),
                (3, 4),
                (5, 6),
                (7, 8),
                (0, 9, 10),
            ),
            "group_sizes": (2, 2, 2, 2, 3),
            "groups": 5,
            "max_group_paths": 3,
            "launches": 1,
            "physical_grid_z": (1,),
            "physical_programs": 1,
            "node_updates": 32,
            "critical_node_steps": 32,
            "live_state_tiles": 2,
            "nominal_register_fp32_values_per_cta": 4096,
            "nominal_register_fp32_values_per_thread": 16,
            "state_export_writes": 0,
            "state_parent_reads": 0,
            "single_writer_nodes": 32,
        },
    }

    namespace["_fr13_fixed32_observed_gdn"](
        "gdn.0",
        0,
        1,
        32,
        32,
        32,
        32,
        32,
        32,
        (32, 32),
        (32, 32),
        runtime_state,
    )

    assert event["gdn_path_programs"] == 12
    assert event["gdn_physical_route"] == "fixed32_single_launch_tree"
    assert event["gdn_physical_programs"] == 1
    assert event["gdn_physical_grid_z"] == (1,)
    assert event["gdn_level1_parent_loads"] == 0
    assert event["gdn_critical_path"] == 12
    assert event["gdn_physical_critical_path"] == 32
