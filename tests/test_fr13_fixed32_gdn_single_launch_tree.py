from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    REPO / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)
PATCHER_PATH = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
VERIFIER_PATH = REPO / "scripts" / "fr13_gdn_single_launch_live_verdict.py"
B1_RUNNER_PATH = REPO / "scripts" / "fr13_run_b1_gdn_single_launch_live_gate.sh"
B4_RUNNER_PATH = REPO / "scripts" / "fr13_run_b4_gdn_single_launch_live_gate.sh"
CORE_RUNNER_PATH = REPO / "scripts" / "fr13_run_gdn_single_launch_live_gate.sh"
RUNTIME_MANIFEST_PATH = REPO / "scripts" / "fr13_runtime_manifest.py"
COMPILE_HARNESS_PATH = REPO / "scripts" / "fr13_compile_gdn_single_launch_sm121.py"
ROOT_LOOP_PATH = (
    REPO
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_gdn_single_launch_root_loop.py"
)
READY_AUDIT_PATH = (
    REPO
    / "results"
    / "fr13_fixed32_gdn_single_launch_root_loop_v1_live_ready_20260802"
)


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
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_IDENTITY_SCHEMA",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_NODE_HELPER",
        "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS",
        "_FR13_FIXED32_PARENT_SHA256",
        "_FR13_FIXED32_ANCESTRY_SHA256",
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


def _load_lifecycle_namespace() -> dict[str, object]:
    constants = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_IDENTITY_SCHEMA",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_NODE_HELPER",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_GATE_ENABLED",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_GATE_ENABLED",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_REAL_EVENT",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_REAL_EVENT",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_PASS",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_PASS",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXACT4_MARKERS",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXACT4_SHA256",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CAPTURE_CONTEXT",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CAPTURES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_GATE_STATE",
        "_FR13_FIXED32_PARENT_SHA256",
        "_FR13_FIXED32_ANCESTRY_SHA256",
    }
    functions = {
        "_fr13_canonical_sha256",
        "_fr13_fixed32_gdn_single_launch_identity",
        "_fr13_fixed32_gdn_single_launch_gate_enabled",
        "_fr13_fixed32_gdn_single_launch_real_event_marker",
        "_fr13_fixed32_gdn_single_launch_validate_pass",
        "fixed32_gdn_single_launch_selector",
        "_fr13_fixed32_gdn_single_launch_emit_pass",
        "fixed32_gdn_single_launch_live_capture_begin",
        "_fr13_fixed32_gdn_single_launch_capture_register",
        "fixed32_gdn_single_launch_live_capture_end",
        "fixed32_gdn_single_launch_live_gate_on_replay",
        "fixed32_gdn_single_launch_live_gate_report",
    }
    tree, _source = _tree_and_source()
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id in constants
                for target in (
                    node.targets if isinstance(node, ast.Assign) else (node.target,)
                )
            )
        )
        or (isinstance(node, ast.FunctionDef) and node.name in functions)
    ]
    namespace = {
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "stat": stat,
        "Path": Path,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )
    namespace["_FR13_FIXED32_MODE"] = "tail6_fixed32"
    namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH"] = True
    return namespace


class _Bytes:
    def __init__(self, value: int):
        self.value = int(value)

    def clone(self):
        return _Bytes(self.value)

    def copy_(self, other) -> None:
        self.value = int(other.value)


def test_depth_first_contract_covers_each_node_and_writer_once() -> None:
    namespace = _load_contract_namespace()
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]
    contract = namespace["_fr13_fixed32_gdn_single_launch_contract"](levels)

    assert contract["candidate"] == "fixed32_gdn_single_launch_root_loop_v1"
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
    assert hashlib.sha256(helper.encode()).hexdigest() == (
        "39734a9dcfaf14c45de0fd35d8e0b12f6c099a8c3850fdc4c2b472dd3229ca6c"
    )
    assert hashlib.sha256(kernel.encode()).hexdigest() == (
        "870fe9943a8e33b7dff6457b4e8524ef85173cbbea4ddd432a3effd9991cd03d"
    )


def test_b1_b4_launchers_use_one_grid_and_reference_forces_incumbent() -> None:
    b1 = _function_source("launch_tree_gdn_prepared")
    b4 = _function_source("launch_tree_gdn_prepared_fixed32_batch")

    assert "_fr13_fixed32_gdn_single_launch_candidate_kernel()[" in b1
    assert "triton.cdiv(dim_v, _path_block_v),\n                    1," in b1
    assert "COUNT_INVOCATION=_count" in b1
    assert "FLAGS_EXPORT=_flags_export" in b1
    assert "and not force_reference_structure" in b1

    assert "_fr13_fixed32_gdn_single_launch_candidate_kernel()[" in b4
    assert "triton.cdiv(dim_v, _block_v),\n                    batch," in b4
    assert "COUNT_INVOCATION=count_invocation" in b4
    assert "FLAGS_EXPORT=flags_export" in b4
    assert "force_reference_structure=True" in b4
    assert b4.index("force_reference_structure=True") > b4.index(
        "def _launch_reference"
    )
    loader = _function_source(
        "_fr13_fixed32_gdn_single_launch_candidate_kernel"
    )
    assert "from . import fr13_gdn_single_launch_root_loop as candidate_module" in loader
    assert "candidate_module.CANDIDATE" in loader
    assert "_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL" in loader


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

    assert 'physical_route = str(executed["route"])' in observer
    assert 'executed = runtime_state.get("executed_gdn")' in observer
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
        "gdn_candidate": None,
        "gdn_physical_launches": 0,
        "gdn_physical_programs": 0,
        "gdn_physical_grid_z": None,
        "gdn_level1_parent_loads": 0,
        "gdn_state_export_writes": 0,
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
            "candidate": "fixed32_gdn_single_launch_root_loop_v1",
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
        "executed_gdn": {
            "route": "fixed32_single_launch_root_loop",
            "candidate": "fixed32_gdn_single_launch_root_loop_v1",
            "physical_launches": 1,
            "physical_programs": 1,
            "physical_grid_z": (1,),
            "physical_recurrence_critical_path": 32,
            "state_export_writes": 0,
            "state_parent_reads": 0,
            "logical_launches": 2,
            "logical_programs": 12,
            "logical_padded_slots": 82,
            "logical_critical_path": 12,
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
    assert event["gdn_physical_route"] == "fixed32_single_launch_root_loop"
    assert event["gdn_candidate"] == "fixed32_gdn_single_launch_root_loop_v1"
    assert event["gdn_physical_launches"] == 1
    assert event["gdn_physical_programs"] == 1
    assert event["gdn_physical_grid_z"] == (1,)
    assert event["gdn_level1_parent_loads"] == 0
    assert event["gdn_state_export_writes"] == 0
    assert event["gdn_critical_path"] == 12
    assert event["gdn_physical_critical_path"] == 32


def _single_launch_identity(namespace: dict[str, object], batch: int) -> dict:
    contract = {
        "schema": namespace[
            "_FR13_FIXED32_GDN_SINGLE_LAUNCH_IDENTITY_SCHEMA"
        ],
        "candidate": namespace[
            "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID"
        ],
        "kernel": namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL"],
        "node_helper": namespace[
            "_FR13_FIXED32_GDN_SINGLE_LAUNCH_NODE_HELPER"
        ],
        "physical_grid_z": (1,),
        "physical_programs": 1,
        "critical_node_steps": 32,
        "state_export_writes": 0,
        "state_parent_reads": 0,
        "single_writer_nodes": 32,
        "parent_sha256": namespace["_FR13_FIXED32_PARENT_SHA256"],
        "ancestry_sha256": namespace["_FR13_FIXED32_ANCESTRY_SHA256"],
        "contract_sha256": "a" * 64,
        "groups_sha256": "b" * 64,
        "execution_sha256": "c" * 64,
    }
    return namespace["_fr13_fixed32_gdn_single_launch_identity"](
        contract,
        batch,
        source_sha256="d" * 64,
        support_source_sha256="e" * 64,
        mode="tail6_fixed32",
    )


def _single_launch_gate_record(namespace, identity, layer_key, batch):
    current = {
        "out": _Bytes(1),
        "export": _Bytes(7),
        "ring_k": _Bytes(1),
        "ring_v": _Bytes(1),
        "ring_a": _Bytes(1),
        "ring_b": _Bytes(1),
        "flags": _Bytes(1),
        "counter": _Bytes(10),
    }

    def snapshot():
        return {name: value.clone() for name, value in current.items()}

    def restore(saved):
        for name, value in saved.items():
            current[name].copy_(value)

    def run_reference():
        for name in ("out", "ring_k", "ring_v", "ring_a", "ring_b", "flags"):
            current[name].value = 1
        current["counter"].value = 11
        current["export"].value = 9
        return {
            "kernel_structure": "fixed32_path",
            "physical_launches": 2 * batch,
            "state_export_writes": 5 * batch,
        }

    def run_candidate():
        for name in ("out", "ring_k", "ring_v", "ring_a", "ring_b", "flags"):
            current[name].value = 1
        current["counter"].value = 11
        return {
            "candidate": namespace[
                "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID"
            ],
            "kernel_structure": namespace[
                "_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL"
            ],
            "identity_sha256": identity["identity_sha256"],
            "physical_launches": 1,
            "state_export_writes": 0,
        }

    return {
        "record": {
            "layer_key": layer_key,
            "identity": identity,
            "snapshot": snapshot,
            "restore": restore,
            "run_reference": run_reference,
            "run_candidate": run_candidate,
            "carrier_nonzero": lambda: True,
            "byte_equal": lambda left, right: left.value == right.value,
        },
        "current": current,
    }


@pytest.mark.parametrize("batch", (1, 4))
def test_authenticated_gate_restores_reference_and_emits_only_complete_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
) -> None:
    namespace = _load_lifecycle_namespace()
    identity = _single_launch_identity(namespace, batch)
    enabled = tmp_path / "enabled"
    marker_path = tmp_path / "real-event.arm"
    pass_path = tmp_path / "live-pass.json"
    namespace[f"_FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_GATE_ENABLED"] = str(
        enabled
    )
    namespace[f"_FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_REAL_EVENT"] = str(
        marker_path
    )
    namespace[f"_FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_PASS"] = str(pass_path)
    monkeypatch.setenv(
        f"FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_BYTE_AB", "1"
    )
    monkeypatch.setenv(
        f"FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_REAL_EVENT_PATH",
        str(marker_path),
    )
    monkeypatch.setenv(
        f"FR13_FIXED32_GDN_SINGLE_LAUNCH_B{batch}_PASS_PATH", str(pass_path)
    )
    graph_id = 700 + batch
    signature = "e" * 64
    namespace["fixed32_gdn_single_launch_live_capture_begin"](graph_id, batch)
    carriers = []
    for layer in range(48):
        carrier = _single_launch_gate_record(namespace, identity, layer + 1, batch)
        carriers.append(carrier)
        namespace["_fr13_fixed32_gdn_single_launch_capture_register"](
            carrier["record"]
        )
    namespace["fixed32_gdn_single_launch_live_capture_end"](
        graph_id, batch, signature, 48
    )

    markers = (
        ("swe_verified:astropy__astropy-12907",)
        if batch == 1
        else namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_EXACT4_MARKERS"]
    )
    for index, marker in enumerate(markers):
        marker_path.write_text(marker + "\n", encoding="ascii")
        report = namespace["fixed32_gdn_single_launch_live_gate_on_replay"](
            graph_id, signature, batch, 48
        )
        assert report["comparisons"] == 48 * 7
        assert report["reference_served"] is True
        assert report["candidate_export_baseline_unchanged"] is True
        assert report["state_restored"] is True
        assert pass_path.exists() is (index == len(markers) - 1)
        assert all(
            carrier["current"][name].value == value
            for carrier in carriers
            for name, value in {
                "out": 1,
                "export": 7,
                "ring_k": 1,
                "ring_v": 1,
                "ring_a": 1,
                "ring_b": 1,
                "flags": 1,
                "counter": 10,
            }.items()
        )

    payload = namespace["_fr13_fixed32_gdn_single_launch_validate_pass"](
        batch, identity, pass_path=str(pass_path)
    )
    assert payload["candidate_physical_launches_per_layer"] == 1
    assert payload["candidate_state_export_writes"] == 0


def test_static_single_launch_pass_cannot_authorize_root_loop(
    tmp_path: Path,
) -> None:
    namespace = _load_lifecycle_namespace()
    identity = _single_launch_identity(namespace, 1)
    legacy = tmp_path / "legacy-pass.json"
    legacy.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.gdn_single_launch.b1_live_pass.v2",
                "status": "pass",
                "candidate": "fixed32_gdn_single_launch_tree_v2",
            }
        )
        + "\n",
        encoding="ascii",
    )
    with pytest.raises(RuntimeError, match="identity/contract is invalid"):
        namespace["_fr13_fixed32_gdn_single_launch_validate_pass"](
            1, identity, pass_path=str(legacy)
        )


def test_single_launch_gate_restores_baseline_on_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _load_lifecycle_namespace()
    identity = _single_launch_identity(namespace, 1)
    marker_path = tmp_path / "real-event.arm"
    pass_path = tmp_path / "live-pass.json"
    marker_path.write_text(
        "swe_verified:astropy__astropy-12907\n", encoding="ascii"
    )
    monkeypatch.setenv("FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_BYTE_AB", "1")
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_REAL_EVENT_PATH", str(marker_path)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_PASS_PATH", str(pass_path)
    )
    namespace["fixed32_gdn_single_launch_live_capture_begin"](801, 1)
    carriers = []
    for layer in range(48):
        carrier = _single_launch_gate_record(namespace, identity, layer + 1, 1)
        carriers.append(carrier)
        if layer == 7:
            original = carrier["record"]["run_candidate"]

            def mismatch(original=original, current=carrier["current"]):
                metadata = original()
                current["out"].value = 2
                return metadata

            carrier["record"]["run_candidate"] = mismatch
        namespace["_fr13_fixed32_gdn_single_launch_capture_register"](
            carrier["record"]
        )
    namespace["fixed32_gdn_single_launch_live_capture_end"](
        801, 1, "f" * 64, 48
    )
    with pytest.raises(RuntimeError, match="byte mismatch"):
        namespace["fixed32_gdn_single_launch_live_gate_on_replay"](
            801, "f" * 64, 1, 48
        )
    assert not pass_path.exists()
    assert all(
        carrier["current"]["out"].value == 1
        and carrier["current"]["export"].value == 7
        and carrier["current"]["counter"].value == 10
        for carrier in carriers
    )


def test_b4_launcher_binds_candidate_to_new_selector_and_graph_gate() -> None:
    source = _function_source("launch_tree_gdn_prepared_fixed32_batch")
    assert 'selector == "single_launch_graph_capture"' in source
    assert 'selector == "single_launch_production"' in source
    assert "_single_candidate=True" in source
    assert '"physical_launches": 1' in source
    assert '"state_export_writes": 0' in source
    assert "_fr13_fixed32_gdn_single_launch_capture_register" in source
    assert "_launch_reference(collect_export=False)" in source


def test_patcher_binds_single_launch_gate_and_actual_route_census() -> None:
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    assert "fixed32_gdn_single_launch_live_capture_begin(identity, batch)" in patcher
    assert "fixed32_gdn_single_launch_live_capture_end(" in patcher
    assert "fixed32_gdn_single_launch_live_gate_on_replay(" in patcher
    assert '"executed_gdn": _fr13_f32_scan_state.get(' in patcher
    assert '"legacy_structure_semantics": "logical_fixed32_path_equivalent"' in patcher
    assert '"logical_launches": int(work["gdn_launches"])' in patcher
    assert '"physical_launches_per_layer": (' in patcher
    assert '"state_export_writes_per_layer": (' in patcher


@pytest.mark.parametrize("batch", (1, 4))
@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
def test_live_verifier_recomputes_exact_source_identity(batch: int, mode: str) -> None:
    spec = importlib.util.spec_from_file_location(
        "fr13_gdn_single_launch_live_verdict",
        VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    namespace = _load_lifecycle_namespace()
    contract = {
        "schema": namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_IDENTITY_SCHEMA"],
        "candidate": namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID"],
        "kernel": namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_KERNEL"],
        "node_helper": namespace["_FR13_FIXED32_GDN_SINGLE_LAUNCH_NODE_HELPER"],
        "physical_grid_z": (1,),
        "physical_programs": 1,
        "critical_node_steps": 32,
        "state_export_writes": 0,
        "state_parent_reads": 0,
        "single_writer_nodes": 32,
        "parent_sha256": namespace["_FR13_FIXED32_PARENT_SHA256"],
        "ancestry_sha256": namespace["_FR13_FIXED32_ANCESTRY_SHA256"],
        "contract_sha256": verifier.CONTRACT_SHA256,
        "groups_sha256": verifier.GROUPS_SHA256,
        "execution_sha256": verifier.EXECUTION_SHA256,
    }
    source_sha256 = hashlib.sha256(ROOT_LOOP_PATH.read_bytes()).hexdigest()
    support_source_sha256 = hashlib.sha256(KERNEL_PATH.read_bytes()).hexdigest()
    namespace["_FR13_FIXED32_MODE"] = mode
    source_identity = namespace["_fr13_fixed32_gdn_single_launch_identity"](
        contract,
        batch,
        source_sha256=source_sha256,
        support_source_sha256=support_source_sha256,
        mode=mode,
    )
    verifier_identity = verifier.single_launch_identity(
        batch_size=batch,
        mode=mode,
        source_sha256=source_sha256,
        support_source_sha256=support_source_sha256,
    )
    assert verifier_identity == source_identity


def test_live_verifier_rejects_mutated_resource_audit(
    tmp_path: Path,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "fr13_gdn_single_launch_live_verdict_checksums",
        VERIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    for filename in verifier.RESOURCE_AUDIT_FILES:
        (tmp_path / filename).write_bytes(f"{filename}\n".encode("ascii"))
    checksum_text = "".join(
        f"{hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()}  {filename}\n"
        for filename in verifier.RESOURCE_AUDIT_FILES
    )
    (tmp_path / "SHA256SUMS").write_text(checksum_text, encoding="ascii")

    expected = hashlib.sha256(checksum_text.encode("ascii")).hexdigest()
    assert verifier._validate_checksum_manifest(tmp_path) == expected
    (tmp_path / "verification.json").write_text("mutated\n", encoding="ascii")
    with pytest.raises(verifier.VerdictError, match="checksum failed"):
        verifier._validate_checksum_manifest(tmp_path)


def test_live_launcher_and_runners_are_reference_served_k64_exact_task_only() -> None:
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    core = CORE_RUNNER_PATH.read_text(encoding="utf-8")
    b1 = B1_RUNNER_PATH.read_text(encoding="utf-8")
    b4 = B4_RUNNER_PATH.read_text(encoding="utf-8")
    verifier = VERIFIER_PATH.read_text(encoding="utf-8")
    runtime_manifest = RUNTIME_MANIFEST_PATH.read_text(encoding="utf-8")
    assert "fr13_fixed32_gdn_single_launch_tree.arm" in launcher
    assert "fr13_fixed32_gdn_single_launch_b1_byte_ab.enabled" in launcher
    assert "fr13_fixed32_gdn_single_launch_b4_byte_ab.enabled" in launcher
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_REAL_EVENT_PATH" in launcher
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_REAL_EVENT_PATH" in launcher
    assert "FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0" in core
    assert "fixed32_gdn_single_launch_root_loop_v1" in core
    assert "--support-source \"$SUPPORT_SOURCE\"" in core
    assert "audit_source_sha256" in core
    assert 'CAPTURE_ONLY=0 ACCEPT_SPEED_PROBE=0 PROBE_ONLY=0' in core
    assert "FR13_DRAFT_VOCAB_K=65536" in core
    assert "FR13_DRAFT_VOCAB_ROOT=1" in core
    assert (
        "STOCK_FA2_SHA256="
        "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d" in core
    )
    assert "CAPTURE_ONLY=0 ACCEPT_SPEED_PROBE=0 PROBE_ONLY=0" in core
    assert 'FR13_FIXED32_GDN_SINGLE_LAUNCH_B1_BYTE_AB="$B1_GATE"' in core
    assert 'FR13_FIXED32_GDN_SINGLE_LAUNCH_B4_BYTE_AB="$B4_GATE"' in core
    assert 'fr13_run_gdn_single_launch_live_gate.sh" b1' in b1
    assert 'fr13_run_gdn_single_launch_live_gate.sh" b4' in b4
    assert "core_runner_path.relative_to(REPO).as_posix()" in verifier
    for path in (
        "scripts/fr13_run_b1_gdn_single_launch_live_gate.sh",
        "scripts/fr13_run_b4_gdn_single_launch_live_gate.sh",
        "scripts/fr13_run_gdn_single_launch_live_gate.sh",
        "scripts/fr13_gdn_single_launch_live_verdict.py",
        "src/lumo_flywheel_serving/fr13_gdn_single_launch_root_loop.py",
        "results/fr13_fixed32_gdn_single_launch_root_loop_v1_live_ready_20260802/SHA256SUMS",
        "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    ):
        assert f'"{path}"' in runtime_manifest


def test_codegen_root_loop_changes_only_the_outer_ordered_loop() -> None:
    production_tree, _production_source = _tree_and_source(KERNEL_PATH)
    candidate_tree, candidate_source = _tree_and_source(ROOT_LOOP_PATH)
    production = copy.deepcopy(
        next(
            node
            for node in production_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_tree_gdn_kernel_fixed32_single_launch"
        )
    )
    candidate = copy.deepcopy(
        next(
            node
            for node in candidate_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name
            == "_tree_gdn_kernel_fixed32_single_launch_root_loop"
        )
    )

    class ReplaceRootStaticRange(ast.NodeTransformer):
        replacements = 0

        def visit_For(self, node: ast.For) -> ast.For:
            self.generic_visit(node)
            call = node.iter
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "static_range"
                and len(call.args) == 2
                and isinstance(call.args[1], ast.Name)
                and call.args[1].id == "ROOT_STEPS"
            ):
                call.func.attr = "range"
                self.replacements += 1
            return node

    transformer = ReplaceRootStaticRange()
    production = transformer.visit(production)
    assert transformer.replacements == 1
    production.name = candidate.name = "normalized_single_launch"
    production.body[0] = candidate.body[0] = ast.Expr(value=ast.Constant(value=""))
    assert ast.dump(production, include_attributes=False) == ast.dump(
        candidate,
        include_attributes=False,
    )

    candidate_kernel = _function_source(
        "_tree_gdn_kernel_fixed32_single_launch_root_loop",
        path=ROOT_LOOP_PATH,
    )
    assert "for root_index in tl.range(0, ROOT_STEPS):" in candidate_kernel
    assert "for member in tl.static_range(0, MAX_GROUP_PATHS):" in candidate_kernel
    assert "for path_offset in tl.range(0, path_len):" in candidate_kernel
    assert candidate_kernel.count("_tree_gdn_fixed32_single_launch_node(") == 2
    assert "state_export" not in candidate_kernel
    assert "fr13_gdn_single_launch_root_loop" not in LAUNCHER_PATH.read_text(
        encoding="utf-8"
    )
    assert "fr13_gdn_single_launch_root_loop.py" in RUNTIME_MANIFEST_PATH.read_text(
        encoding="utf-8"
    )
    assert 'CANDIDATE = "fixed32_gdn_single_launch_root_loop_v1"' in candidate_source


def test_sm121_harness_directly_targets_the_root_loop_candidate() -> None:
    source = COMPILE_HARNESS_PATH.read_text(encoding="utf-8")
    candidate_sha256 = hashlib.sha256(ROOT_LOOP_PATH.read_bytes()).hexdigest()

    assert '"fr13_gdn_single_launch_root_loop.py"' in source
    assert "SUPPORT_PATH = REPO / \"src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py\"" in source
    assert f'"{candidate_sha256}"' in source
    assert "types.ModuleType(package_name)" in source
    assert '"lumo_flywheel_serving.fr13_gdn_single_launch_root_loop"' in source


def test_root_loop_ready_artifact_is_reduced_source_bound_and_not_executed() -> None:
    expected_files = {
        "README.md",
        "SHA256SUMS",
        "codegen.json",
        "manifest.json",
        "resources.tsv",
        "source_hashes.tsv",
        "verification.json",
    }
    assert {path.name for path in READY_AUDIT_PATH.iterdir()} == expected_files
    checksum_rows = {}
    for line in (READY_AUDIT_PATH / "SHA256SUMS").read_text(
        encoding="ascii"
    ).splitlines():
        digest, filename = line.split("  ", 1)
        checksum_rows[filename] = digest
    assert set(checksum_rows) == expected_files - {"SHA256SUMS"}
    for filename, digest in checksum_rows.items():
        assert hashlib.sha256(
            (READY_AUDIT_PATH / filename).read_bytes()
        ).hexdigest() == digest

    manifest = json.loads((READY_AUDIT_PATH / "manifest.json").read_text())
    verification = json.loads(
        (READY_AUDIT_PATH / "verification.json").read_text()
    )
    codegen = json.loads((READY_AUDIT_PATH / "codegen.json").read_text())
    assert manifest["status"] == "READY_NOT_EXECUTED"
    assert manifest["candidate"] == "fixed32_gdn_single_launch_root_loop_v1"
    assert verification["status"] == "READY_NOT_EXECUTED"
    assert verification["checks"]["gpu_kernel_not_executed"] is True
    assert verification["checks"]["duplicate_sm121_build_passed"] is True
    assert codegen["status"] == "PASS_ZERO_SPILL_READY_NOT_EXECUTED"
    assert {variant["batch"] for variant in codegen["variants"]} == {1, 4}
    assert all(
        variant["kernel"]
        == "_tree_gdn_kernel_fixed32_single_launch_root_loop"
        and variant["ctas_per_request"] == 768
        and variant["sass_instructions"] == 1592
        and variant["stack_bytes"] == 0
        and variant["ldl_instructions"] == 0
        and variant["stl_instructions"] == 0
        and variant["call_instructions"] == 0
        for variant in codegen["variants"]
    )
    assert not any(
        path.suffix in {".cubin", ".ptx", ".sass", ".ttir", ".ttgir", ".log"}
        for path in READY_AUDIT_PATH.iterdir()
    )
