from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"


def _load_namespace(
    constants: set[str], functions: set[str]
) -> dict[str, object]:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    namespace = {"hashlib": hashlib, "json": json, "os": os}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            KERNEL,
            "exec",
        ),
        namespace,
    )
    return namespace


def _function_source(name: str) -> str:
    source = KERNEL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    segment = ast.get_source_segment(source, node)
    assert segment is not None
    return segment


def test_prescaled_descriptor_is_exact_reindexing_of_fixed32_paths() -> None:
    namespace = _load_namespace(
        {
            "_FR13_FIXED32_PARENT",
            "_FR13_FIXED32_SUBTREE_LEVELS",
            "_FR13_FIXED32_EXPORT_NODES",
            "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID",
            "_FR13_FIXED32_GDN_DEPTH_FIRST_GROUPS",
        },
        {
            "_fr13_canonical_sha256",
            "_fr13_fixed32_gdn_single_launch_contract",
            "_fr13_fixed32_gdn_prescaled_path_descriptor",
        },
    )
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]
    contract = namespace["_fr13_fixed32_gdn_single_launch_contract"](levels)
    descriptor = namespace[
        "_fr13_fixed32_gdn_prescaled_path_descriptor"
    ](levels, contract)

    assert descriptor["schema"] == "fr13.fixed32.gdn_prescaled_path_base.v1"
    assert descriptor["max_path_len"] == 7
    assert descriptor["path_bases"] == (
        (7, 14, 0),
        (21, 28, 0),
        (35, 42, 0),
        (49, 56, 0),
        (0, 63, 70),
    )
    lengths = descriptor["path_base_lengths"]
    assert len(lengths) == 77
    assert {index: value for index, value in enumerate(lengths) if value} == {
        0: 7,
        7: 5,
        14: 7,
        21: 1,
        28: 1,
        35: 1,
        42: 1,
        49: 1,
        56: 1,
        63: 1,
        70: 1,
    }

    branch_paths = levels[1]
    flat_nodes = tuple(
        node
        for path, _parent in branch_paths
        for node in tuple(path) + (-1,) * (7 - len(path))
    )
    for group_bases, path_indices in zip(
        descriptor["path_bases"],
        contract["branch_path_indices"],
        strict=True,
    ):
        for base, path_index in zip(group_bases, path_indices, strict=False):
            path = tuple(branch_paths[path_index][0])
            assert base == path_index * 7
            assert lengths[base] == len(path)
            assert flat_nodes[base : base + len(path)] == path


def test_prescaled_selector_is_default_off_exact_and_single_launch_bound(
    tmp_path: Path,
) -> None:
    namespace = _load_namespace(
        {"_FR13_FIXED32_GDN_PRESCALED_PATH_BASE_SIDECARS"},
        {"_fr13_resolve_fixed32_gdn_prescaled_path_base"},
    )
    resolve = namespace["_fr13_resolve_fixed32_gdn_prescaled_path_base"]

    assert not resolve(True, environ={}, sidecars=())
    assert not resolve(
        True,
        environ={"FR13_FIXED32_GDN_PRESCALED_PATH_BASE": "0"},
        sidecars=(),
    )
    assert resolve(
        True,
        environ={"FR13_FIXED32_GDN_PRESCALED_PATH_BASE": "1"},
        sidecars=(),
    )
    with pytest.raises(RuntimeError, match="K64/root1 ordered single-launch"):
        resolve(
            False,
            environ={"FR13_FIXED32_GDN_PRESCALED_PATH_BASE": "1"},
            sidecars=(),
        )
    with pytest.raises(RuntimeError, match="exactly 0 or 1"):
        resolve(
            True,
            environ={"FR13_FIXED32_GDN_PRESCALED_PATH_BASE": "true"},
            sidecars=(),
        )

    arm = tmp_path / "prescaled.arm"
    arm.write_text("1\n", encoding="ascii")
    assert resolve(True, environ={}, sidecars=(str(arm),))
    with pytest.raises(RuntimeError, match="agreeing sources"):
        resolve(
            True,
            environ={"FR13_FIXED32_GDN_PRESCALED_PATH_BASE": "0"},
            sidecars=(str(arm),),
        )


def test_kernel_changes_only_descriptor_addressing_and_wires_b1_b4() -> None:
    kernel = _function_source("_tree_gdn_kernel_fixed32_single_launch")
    preseed = _function_source("subtree_preseed")
    b1 = _function_source("launch_tree_gdn_prepared")
    b4 = _function_source("launch_tree_gdn_prepared_fixed32_batch")
    selector = _function_source(
        "_fr13_resolve_fixed32_gdn_prescaled_path_base"
    )
    path_args = _function_source(
        "_fr13_fixed32_gdn_single_launch_path_args"
    )

    assert "PRESCALED_PATH_BASE: tl.constexpr = False" in kernel
    assert "path_base = path_index * MAX_PATH_LEN" in kernel
    assert "path_base = path_index" in kernel
    assert "branch_nodes + path_base + path_offset" in kernel
    assert "for path_offset in tl.range(0, path_len):" in kernel
    assert kernel.count("_tree_gdn_fixed32_single_launch_node(") == 2
    for rejected in (
        "STATIC_ROOT_NODE",
        "STATIC_GROUP_COUNT",
        "PACKED_PATH_META",
        "STATIC_PATH_LENGTH",
        "PACKED_PATH_RECORD",
        "PACKED_GROUP_COUNT",
    ):
        assert rejected not in kernel

    assert 'raw_env = env.get("FR13_FIXED32_GDN_PRESCALED_PATH_BASE")' in selector
    assert "if enabled and not single_launch_available:" in selector
    assert "no fallback is permitted" in path_args
    assert '"prescaled_path_bases": torch.tensor(' in preseed
    assert '"prescaled_path_base_lengths": torch.tensor(' in preseed
    for launch in (b1, b4):
        assert "_fr13_fixed32_gdn_single_launch_path_args(" in launch
        assert "PRESCALED_PATH_BASE=" in launch
        assert '"prescaled_path_base": bool(' in launch

