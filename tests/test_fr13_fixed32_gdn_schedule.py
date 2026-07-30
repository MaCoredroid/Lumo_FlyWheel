from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


KERNEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "lumo_flywheel_serving"
    / "fr10_gdn_tree_kernel.py"
)
CONSTANTS = {
    "_FR13_FIXED32_PARENT",
    "_FR13_FIXED32_SUBTREE_LEVELS",
    "_FR13_FIXED32_PARENT_SHA256",
    "_FR13_FIXED32_ANCESTRY_SHA256",
    "_FR13_FIXED32_LEVELS_SHA256",
    "_FR13_FIXED32_COVERAGE_SHA256",
}
FUNCTIONS = {
    "_fr13_canonical_sha256",
    "_fr13_tree_ancestry",
    "_fr13_fixed32_schedule_contract",
}


def _load_schedule_contract():
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANTS
            for target in node.targets
        )
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    ]
    found_constants = {
        target.id
        for node in assignments
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert found_constants == CONSTANTS
    assert {node.name for node in definitions} == FUNCTIONS

    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *assignments,
            *definitions,
        ],
        type_ignores=[],
    )
    namespace = {"hashlib": hashlib, "json": json}
    exec(compile(ast.fix_missing_locations(module), KERNEL_PATH, "exec"), namespace)
    return namespace


def test_fixed32_gdn_schedule_uses_parent_bit_union() -> None:
    namespace = _load_schedule_contract()
    levels = namespace["_FR13_FIXED32_SUBTREE_LEVELS"]

    contract = namespace["_fr13_fixed32_schedule_contract"](levels)

    assert contract == {
        "path_counts": (1, 11),
        "max_lengths": (5, 7),
        "launches": 2,
        "programs": 12,
        "padded_slots": 82,
        "critical": 12,
        "export_or_mask": 16915,
        "parent_sha256": namespace["_FR13_FIXED32_PARENT_SHA256"],
        "ancestry_sha256": namespace["_FR13_FIXED32_ANCESTRY_SHA256"],
        "levels_sha256": namespace["_FR13_FIXED32_LEVELS_SHA256"],
        "coverage_sha256": namespace["_FR13_FIXED32_COVERAGE_SHA256"],
    }

    repeated_parent_sum = sum(
        1 << parent
        for level in levels[1:]
        for _path, parent in level
    )
    assert repeated_parent_sum == 50214
    assert repeated_parent_sum != contract["export_or_mask"]
