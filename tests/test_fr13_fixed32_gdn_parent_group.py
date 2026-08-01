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


def _load_parent_group_contract() -> dict[str, object]:
    constants = {
        "_FR13_FIXED32_PARENT",
        "_FR13_FIXED32_SUBTREE_LEVELS",
        "_FR13_FIXED32_EXPORT_NODES",
        "_FR13_FIXED32_GDN_PARENT_GROUP_CANDIDATE_ID",
        "_FR13_FIXED32_GDN_LEVEL1_PARENT_GROUPS",
        "_FR13_FIXED32_COVERAGE_SHA256",
    }
    functions = {
        "_fr13_canonical_sha256",
        "_fr13_fixed32_gdn_parent_group_contract",
        "_fr13_fixed32_gdn_physical_execution",
    }
    tree, _source = _tree_and_source()
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in constants
            for target in node.targets
        )
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    module = ast.Module(body=[*assignments, *definitions], type_ignores=[])
    namespace = {"hashlib": hashlib, "json": json}
    exec(
        compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"),
        namespace,
    )
    assert constants | functions <= namespace.keys()
    return namespace


def _load_selector() -> dict[str, object]:
    constants = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_PARENT_GROUP_SIDECARS",
    }
    function = "_fr13_resolve_fixed32_gdn_parent_group"
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
        compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                KERNEL_PATH, "exec"),
        namespace,
    )
    return namespace


def test_parent_group_descriptor_covers_paths_nodes_and_writers_once() -> None:
    namespace = _load_parent_group_contract()
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]
    contract = namespace["_fr13_fixed32_gdn_parent_group_contract"](levels)

    assert contract["parent_nodes"] == (14, 0, 1, 4, 9)
    assert contract["parent_slots"] == (4, 0, 1, 2, 3)
    assert contract["path_indices"] == (
        (0, 9, 10),
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    )
    assert contract["group_sizes"] == (3, 2, 2, 2, 2)
    assert contract["simd_width"] == 4
    assert contract["member_execution"] == "parallel_simd"
    assert contract["group_node_counts"] == (9, 12, 2, 2, 2)
    assert contract["group_max_path_lengths"] == (7, 7, 1, 1, 1)
    assert contract["physical_level_max_steps"] == (5, 7)
    assert contract["physical_critical_path"] == 12
    assert contract["physical_grid_z"] == (1, 5)
    assert contract["physical_programs"] == 6
    assert contract["logical_programs"] == 12
    assert contract["level1_parent_loads"] == 5
    assert contract["reference_level1_parent_loads"] == 11

    level1 = levels[1]
    covered_paths = [
        index
        for indices in contract["path_indices"]
        for index in indices
    ]
    assert sorted(covered_paths) == list(range(11))
    for parent, indices in zip(
        contract["parent_nodes"], contract["path_indices"], strict=True
    ):
        assert all(level1[index][1] == parent for index in indices)

    output_writers = [
        node
        for level in levels
        for path, _parent in level
        for node in path
    ]
    ring_writers = list(output_writers)
    assert sorted(output_writers) == list(range(32))
    assert sorted(ring_writers) == list(range(32))
    assert all(output_writers.count(node) == 1 for node in range(32))
    assert all(ring_writers.count(node) == 1 for node in range(32))
    assert contract["single_writer_nodes"] == 32


def test_parent_group_contract_rejects_duplicate_or_wrong_parent() -> None:
    namespace = _load_parent_group_contract()
    validate = namespace["_fr13_fixed32_gdn_parent_group_contract"]
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]

    with pytest.raises(RuntimeError, match="path coverage drift"):
        validate(
            levels,
            groups=((14, (0, 0, 10)), (0, (1, 2)), (1, (3, 4)),
                    (4, (5, 6)), (9, (7, 8))),
        )
    with pytest.raises(RuntimeError, match="parent/path mismatch"):
        validate(
            levels,
            groups=((0, (0, 9, 10)), (14, (1, 2)), (1, (3, 4)),
                    (4, (5, 6)), (9, (7, 8))),
        )


def test_parent_group_selector_is_default_off_and_fail_closed(tmp_path) -> None:
    namespace = _load_selector()
    resolve = namespace["_fr13_resolve_fixed32_gdn_parent_group"]
    kwargs = {
        "fixed32_mode": "hydra27_fixed32",
        "sidecars": (),
        "geom_override": {"BV": 8},
    }

    assert resolve(environ={}, **kwargs) is False
    assert resolve(
        environ={"FR13_FIXED32_GDN_PARENT_GROUP": "1"}, **kwargs
    ) is True
    with pytest.raises(RuntimeError, match="must be exactly 1"):
        resolve(
            environ={"FR13_FIXED32_GDN_PARENT_GROUP": "0"}, **kwargs
        )
    with pytest.raises(RuntimeError, match="requires an exact fixed32 mode"):
        resolve(
            None,
            environ={"FR13_FIXED32_GDN_PARENT_GROUP": "1"},
            sidecars=(),
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="pinned exactly"):
        resolve(
            environ={"FR13_FIXED32_GDN_PARENT_GROUP": "1"},
            sidecars=(),
            geom_override={"BV": 16},
            fixed32_mode="hydra27_fixed32",
        )

    sidecar = tmp_path / "parent-group.arm"
    sidecar.write_text("1\n", encoding="ascii")
    assert resolve(environ={}, sidecars=(sidecar,),
                   fixed32_mode="tail6_fixed32",
                   geom_override={"BV": 8}) is True


def test_parent_group_kernel_and_launchers_keep_single_writer_surfaces() -> None:
    kernel = _function_source("_tree_gdn_path_kernel_fixed32_parent_group")
    group_step = _function_source("_gdn_group_node_step")
    b1_launcher = _function_source("launch_tree_gdn_prepared")
    batched_launcher = _function_source(
        "launch_tree_gdn_prepared_fixed32_batch"
    )

    assert "offs_member = tl.arange(0, SIMD_WIDTH)[:, None]" in kernel
    assert "for member in tl.static_range" not in kernel
    assert "for i in tl.static_range(0, MAX_PATH_LEN)" in kernel
    assert kernel.index("parent_state = tl.load(") < kernel.index(
        "for i in tl.static_range"
    )
    assert "state_i = parent_state[None, :, :] + tl.zeros(" in kernel
    assert "_gdn_group_node_step(" in kernel
    assert "state_i * b_k[:, None, :]" in group_step
    assert "axis=2" in group_step
    assert "b_v[:, :, None] * b_k[:, None, :]" in group_step
    assert "invocation_counter" not in kernel
    assert "flags_ptr" not in kernel
    assert "tl.store(\n                state_export" not in kernel
    assert "mask=n_ok & (pid_v == 0)" in kernel
    assert "(pid_vh % head_group == 0)" in kernel

    assert "if _parent_group is not None and _li == 1:" in b1_launcher
    assert "COMPACT_EXPORT=False" in b1_launcher
    assert "BATCH_SIZE=1" in b1_launcher
    assert 'SIMD_WIDTH=int(_group_contract["simd_width"])' in b1_launcher
    assert "COUNT_INVOCATION=_count and (_li == 0)" in b1_launcher
    assert "FLAGS_EXPORT=_flags_export and (_li == 0)" in b1_launcher
    assert "if parent_group is not None and level_index == 1:" in batched_launcher
    assert "batch * int(group_contract[\"groups\"])" in batched_launcher
    assert "COMPACT_EXPORT=True" in batched_launcher
    assert "force_incumbent_parent_group=True" in batched_launcher
    assert 'SIMD_WIDTH=int(group_contract["simd_width"])' in batched_launcher
    assert "COUNT_INVOCATION=count_invocation and level_index == 0" in (
        batched_launcher
    )
    assert "FLAGS_EXPORT=flags_export and level_index == 0" in batched_launcher


def test_parent_group_physical_execution_distinguishes_b1_and_b4() -> None:
    namespace = _load_parent_group_contract()
    describe = namespace["_fr13_fixed32_gdn_physical_execution"]

    b1 = describe(parent_group=True, batch_size=1, batched=False)
    assert b1["grid_z_per_request"] == (1, 5)
    assert b1["event_grid_z"] == (1, 5)
    assert b1["physical_launches_per_layer"] == 2
    assert b1["physical_critical_path"] == 12

    b4 = describe(parent_group=True, batch_size=4, batched=True)
    assert b4["grid_z_per_request"] == (1, 5)
    assert b4["event_grid_z"] == (4, 20)
    assert b4["physical_launches_per_layer"] == 2
    assert b4["programs_per_layer"] == 24

    b4_reference = describe(parent_group=False, batch_size=4, batched=False)
    assert b4_reference["event_grid_z"] == (1, 11)
    assert b4_reference["launch_repetitions"] == 4
    assert b4_reference["physical_launches_per_layer"] == 8
    assert b4_reference["physical_critical_path"] == 12


def test_parent_group_b1_gate_is_full_surface_and_reference_served() -> None:
    launcher = _function_source("launch_tree_gdn_prepared")
    source = KERNEL_PATH.read_text(encoding="utf-8")

    assert "force_incumbent_parent_group: bool = False" in launcher
    assert "_use_parent_group=False" in launcher
    assert "_use_parent_group=True" in launcher
    assert '"export": st["export"].clone()' in launcher
    for surface in (
        "output",
        "export",
        "ring_k",
        "ring_v",
        "ring_a",
        "ring_b",
        "flags",
        "counter",
    ):
        assert f'"{surface}"' in source
    assert "_parent_group_gate_restore(reference, gate_counter)" in launcher
    assert "production_authorized=0" in launcher
    assert "parent grouping and parent-gather selfcheck" in source


@pytest.mark.parametrize("batch", [1, 4])
def test_parent_group_static_work_reduction(batch: int) -> None:
    layers = 48
    value_heads = 48
    dim_v = 128
    dim_k = 128
    block_v = 8
    parent_reads_saved = 11 - 5
    state_tile_bytes = value_heads * dim_v * dim_k * 4
    ctas_saved = (
        batch
        * layers
        * value_heads
        * (dim_v // block_v)
        * parent_reads_saved
    )
    bytes_saved = batch * layers * parent_reads_saved * state_tile_bytes

    assert state_tile_bytes == 3_145_728
    if batch == 1:
        assert ctas_saved == 221_184
        assert bytes_saved == 905_969_664
    else:
        assert ctas_saved == 884_736
        assert bytes_saved == 3_623_878_656


def test_parent_group_observer_preserves_logical_contract() -> None:
    observer = _observed_runtime_function_source(
        "_fr13_fixed32_observed_gdn"
    )
    validator = _observed_runtime_function_source(
        "_fr13_fixed32_validate_forward_work"
    )
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert 'physical_route == "fixed32_parent_group_simd"' in observer
    assert '"physical_critical_path": int(' in observer
    assert '"physical_grid_z": (1, 5)' in observer
    assert 'normalized_contract["programs"]' in observer
    assert 'normalized_contract["padded_slots"]' in observer
    assert '"gdn_path_programs": expected_gdn_calls * 12' in validator
    assert '"gdn_padded_slots": expected_gdn_calls * 82' in validator
    assert "expected_physical_programs_per_scan = 6" in validator
    assert "expected_level1_parent_loads_per_scan = 5" in validator
    assert "expected_physical_critical_path = 12" in validator
    assert "expected_physical_event_grid_z" in validator
    assert '"fixed32_parent_group_contract": (' in patcher
    assert '"physical_route": work["gdn_physical_route"]' in patcher


def test_parent_group_observer_records_physical_work_separately() -> None:
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
        "gdn_physical_batched": None,
        "gdn_physical_programs": 0,
        "gdn_physical_grid_z_per_request": None,
        "gdn_physical_event_grid_z": None,
        "gdn_physical_launch_repetitions": None,
        "gdn_physical_launches_per_layer": None,
        "gdn_physical_launches": 0,
        "gdn_level1_parent_loads": 0,
        "gdn_single_writer_nodes": 0,
        "gdn_physical_level_max_steps": None,
        "gdn_physical_critical_path": None,
        "gdn_nodes": 0,
        "gdn_critical_path": None,
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
        "fixed32_parent_group_contract": {
            "candidate": "fixed32_gdn_parent_group_simd_v2",
            "parent_nodes": (14, 0, 1, 4, 9),
            "parent_slots": (4, 0, 1, 2, 3),
            "path_indices": (
                (0, 9, 10),
                (1, 2),
                (3, 4),
                (5, 6),
                (7, 8),
            ),
            "group_sizes": (3, 2, 2, 2, 2),
            "groups": 5,
            "max_group_paths": 3,
            "simd_width": 4,
            "member_execution": "parallel_simd",
            "group_node_counts": (9, 12, 2, 2, 2),
            "group_max_path_lengths": (7, 7, 1, 1, 1),
            "physical_level_max_steps": (5, 7),
            "physical_critical_path": 12,
            "logical_path_counts": (1, 11),
            "physical_grid_z": (1, 5),
            "logical_programs": 12,
            "physical_programs": 6,
            "level1_parent_loads": 5,
            "reference_level1_parent_loads": 11,
            "single_writer_nodes": 32,
        },
        "gdn_physical_execution": {
            "route": "fixed32_parent_group_simd",
            "batched": False,
            "batch_size": 1,
            "grid_z_per_request": (1, 5),
            "event_grid_z": (1, 5),
            "launch_repetitions": 1,
            "physical_launches_per_layer": 2,
            "programs_per_request": 6,
            "programs_per_layer": 6,
            "level1_parent_loads_per_request": 5,
            "single_writer_nodes_per_request": 32,
            "logical_critical_path": 12,
            "physical_level_max_steps": (5, 7),
            "physical_critical_path": 12,
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
    assert event["gdn_padded_slots"] == 82
    assert event["gdn_physical_route"] == "fixed32_parent_group_simd"
    assert event["gdn_physical_programs"] == 6
    assert event["gdn_physical_grid_z_per_request"] == (1, 5)
    assert event["gdn_physical_event_grid_z"] == (1, 5)
    assert event["gdn_physical_launches"] == 2
    assert event["gdn_level1_parent_loads"] == 5
    assert event["gdn_single_writer_nodes"] == 32
    assert event["gdn_physical_level_max_steps"] == (5, 7)
    assert event["gdn_physical_critical_path"] == 12
    assert event["gdn_critical_path"] == 12
