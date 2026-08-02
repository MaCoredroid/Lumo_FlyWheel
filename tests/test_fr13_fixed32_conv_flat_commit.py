from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
)
SOURCE = SOURCE_PATH.read_text()
TREE = ast.parse(SOURCE)


def _function_text(name: str) -> str:
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    return ast.get_source_segment(SOURCE, node) or ""


def _constant(name: str):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name
                for target in item.targets)
    )
    return ast.literal_eval(node.value)


def _resolver():
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_fr13_resolve_fixed32_conv_flat_commit"
    )
    namespace = {
        "os": os,
        "_FR13_FIXED32_CONV_FLAT_COMMIT_ENV": (
            "FR13_FIXED32_CONV_FLAT_COMMIT"
        ),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE_PATH, "exec"),
         namespace)
    return namespace[node.name]


def test_flat_commit_selector_is_default_off_and_fail_closed() -> None:
    resolve = _resolver()
    assert resolve(environ={}) is False
    assert resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": ""}) is False
    assert resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "0"}) is False
    assert (
        resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "diagnostic"})
        is True
    )
    with pytest.raises(RuntimeError, match="must be unset, 0, or diagnostic"):
        resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "1"})


def test_flat_kernel_owns_contiguous_row_and_elides_zero_source_loads() -> None:
    body = _function_text("_fr13_fixed32_conv_flat_zeroelide_col0_kernel")
    assert "flat_start = pid_flat * BLOCK" in body
    assert "state_col = flat_start // CONV_C" in body
    assert "channel = flat_start - state_col * CONV_C" in body
    assert "flat = flat_start + tl.arange(0, BLOCK)" in body
    assert "row_mask = flat < (CONV_C * CONV_L)" in body
    assert "live_col = state_col < LIVE_SOURCE_COLS" in body
    assert "state_src + leaf_node * CONV_L + state_col" in body
    assert "mask=live_col" in body
    assert "mask=row_mask & live_col" in body
    assert "other=0.0" in body
    assert "bank + dst_row.to(tl.int64) * bank_row_stride + flat" in body
    assert "tl.static_range" not in body


def test_flat_route_is_preseeded_once_and_incumbent_remains_fallback() -> None:
    preseed = _function_text("preseed_fixed32_conv_col0_pregather")
    launch = _function_text("launch_fixed32_conv_commit_to_col0")
    assert "flat_commit = _fr13_resolve_fixed32_conv_flat_commit()" in preseed
    assert "state_src[:,3:]==35" in preseed
    assert '"commit_flat_zeroelide": flat_commit' in preseed
    assert 'if state["commit_flat_zeroelide"]:' in launch
    assert "_fr13_fixed32_conv_flat_zeroelide_col0_kernel[grid]" in launch
    assert "_fr13_fixed32_conv_direct_col0_kernel[grid]" in launch
    assert "_fr13_resolve_fixed32_conv_flat_commit" not in launch


def test_flat_contract_is_bound_to_deployed_geometry() -> None:
    assert _constant("_FR13_FIXED32_CONV_FLAT_C") == 10_240
    assert _constant("_FR13_FIXED32_CONV_FLAT_L") == 34
    assert _constant("_FR13_FIXED32_CONV_FLAT_SOURCE_ROWS") == 36
    assert _constant("_FR13_FIXED32_CONV_FLAT_LIVE_SOURCE_COLS") == 3
    assert (
        _constant("_FR13_FIXED32_CONV_FLAT_COMMIT_ROUTE")
        == "fixed32_flat_zeroelide_source_col0"
    )
