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


def _tree_and_source() -> tuple[ast.Module, str]:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    return ast.parse(source), source


def _function_source(name: str) -> str:
    tree, source = _tree_and_source()
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
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
    return namespace


def _load_resolver_namespace() -> dict[str, object]:
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


def test_contract_is_exact_fixed32_single_writer_ordered_schedule() -> None:
    namespace = _load_contract_namespace()
    contract = namespace["_fr13_fixed32_gdn_single_launch_contract"](
        namespace["_FR13_FIXED32_SUBTREE_LEVELS"]
    )

    assert contract["candidate"] == "fixed32_gdn_single_launch_tree_v2"
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
    assert contract["physical_grid_z"] == (1,)
    assert contract["node_updates"] == 32
    assert contract["critical_node_steps"] == 32
    assert contract["live_state_tiles"] == 2
    assert contract["state_export_writes"] == 0
    assert contract["state_parent_reads"] == 0
    assert contract["single_writer_nodes"] == 32
    assert contract["outer_root_loop"] == "ordered_tl_range"


def test_contract_rejects_parent_and_coverage_drift() -> None:
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


def test_selector_is_default_off_and_exact_k64_root1_only() -> None:
    namespace = _load_resolver_namespace()
    resolve = namespace["_fr13_resolve_fixed32_gdn_single_launch"]
    exact = {
        "FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE": "1",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_ROOT": "1",
    }

    assert not resolve(
        "hydra27_fixed32", environ={}, sidecars=(), geom_override={"BV": 8}
    )
    assert resolve(
        "hydra27_fixed32",
        environ=exact,
        sidecars=(),
        geom_override={"BV": 8},
    )
    for key, value, message in (
        ("FR13_DRAFT_VOCAB_K", "32768", "K64/root1"),
        ("FR13_DRAFT_VOCAB_ROOT", "0", "K64/root1"),
        ("FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE", "true", "exactly 1"),
    ):
        drift = dict(exact)
        drift[key] = value
        with pytest.raises(RuntimeError, match=message):
            resolve(
                "hydra27_fixed32",
                environ=drift,
                sidecars=(),
                geom_override={"BV": 8},
            )
    with pytest.raises(RuntimeError, match="geometry pinned exactly"):
        resolve(
            "hydra27_fixed32",
            environ=exact,
            sidecars=(),
            geom_override={"BV": 16},
        )
    with pytest.raises(RuntimeError, match="exact fixed32 mode"):
        resolve(
            None,
            environ=exact,
            sidecars=(),
            geom_override={"BV": 8},
        )


def test_selector_sidecars_must_agree_and_remain_k64_bound(
    tmp_path: Path,
) -> None:
    namespace = _load_resolver_namespace()
    resolve = namespace["_fr13_resolve_fixed32_gdn_single_launch"]
    arm = tmp_path / "single.arm"
    arm.write_text("1\n", encoding="ascii")
    exact = {
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_ROOT": "1",
    }
    assert resolve(
        "tail6_fixed32",
        environ=exact,
        sidecars=(str(arm),),
        geom_override={"BV": 8},
    )
    arm.write_text("0\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="exactly 1"):
        resolve(
            "tail6_fixed32",
            environ=exact,
            sidecars=(str(arm),),
            geom_override={"BV": 8},
        )


def test_kernel_keeps_only_outer_root_loop_dynamic_and_ordered() -> None:
    kernel = _function_source("_tree_gdn_kernel_fixed32_single_launch")
    helper = _function_source("_tree_gdn_fixed32_single_launch_node")

    assert "for root_index in tl.range(0, ROOT_STEPS):" in kernel
    assert "tl.static_range(0, ROOT_STEPS)" not in kernel
    assert "for member in tl.static_range(0, MAX_GROUP_PATHS):" in kernel
    assert "for path_offset in tl.range(0, path_len):" in kernel
    assert kernel.count("_tree_gdn_fixed32_single_launch_node(") == 2
    assert "state_export" not in kernel
    assert "_gdn_node_step(" in helper


def test_b1_b4_dispatch_is_single_launch_and_fail_closed() -> None:
    b1 = _function_source("launch_tree_gdn_prepared")
    b4 = _function_source("launch_tree_gdn_prepared_fixed32_batch")
    selector = _function_source("fixed32_batch_gdn_selector")
    preseed = _function_source("subtree_preseed")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")

    assert "_tree_gdn_kernel_fixed32_single_launch[" in b1
    assert "exact K64/root1 BV8 B1" in b1
    assert '"physical_launches": 1' in b1
    assert '"state_export_writes": 0' in b1
    assert "_tree_gdn_kernel_fixed32_single_launch[" in b4
    assert "exact K64/root1 B4 descriptor" in b4
    assert '"physical_grid_z": (batch,)' in b4
    assert 'elif selector == "single_launch":' in b4
    assert 'return "single_launch" if batch == 4 else None' in selector
    assert "cannot inherit a batched GDN" in selector
    assert '"fixed32_single_launch_contract"' in preseed
    assert '"fixed32_single_launch_contract"' in patcher
    assert '"executed_gdn"' in patcher
