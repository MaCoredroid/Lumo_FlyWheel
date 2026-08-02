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


def _resolver(function_name: str, env_name: str):
    node = next(
        item
        for item in TREE.body
        if isinstance(item, ast.FunctionDef)
        and item.name == function_name
    )
    namespace = {
        "os": os,
        env_name: _constant(env_name),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE_PATH, "exec"),
         namespace)
    return namespace[node.name]


def test_flat_commit_selector_is_default_off_and_fail_closed() -> None:
    resolve = _resolver(
        "_fr13_resolve_fixed32_conv_flat_commit",
        "_FR13_FIXED32_CONV_FLAT_COMMIT_ENV",
    )
    assert resolve(environ={}) is False
    assert resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": ""}) is False
    assert resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "0"}) is False
    assert (
        resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "diagnostic"})
        is True
    )
    with pytest.raises(RuntimeError, match="must be unset, 0, or diagnostic"):
        resolve(environ={"FR13_FIXED32_CONV_FLAT_COMMIT": "1"})


def test_channel_commit_selector_is_default_off_and_fail_closed() -> None:
    resolve = _resolver(
        "_fr13_resolve_fixed32_conv_channel_commit",
        "_FR13_FIXED32_CONV_CHANNEL_COMMIT_ENV",
    )
    assert resolve(environ={}) is False
    assert (
        resolve(environ={"FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT": "0"})
        is False
    )
    assert (
        resolve(
            environ={
                "FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT": "diagnostic"
            }
        )
        is True
    )
    with pytest.raises(RuntimeError, match="must be unset, 0, or diagnostic"):
        resolve(
            environ={"FR13_FIXED32_CONV_CHANNEL_ZEROELIDE_COMMIT": "1"}
        )


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
    kernel_launch = _function_text(
        "_fr13_fixed32_conv_commit_kernel_launch"
    )
    assert "flat_commit = _fr13_resolve_fixed32_conv_flat_commit()" in preseed
    assert "state_src[:,3:]==35" in preseed
    assert '"commit_flat_zeroelide": flat_commit' in preseed
    assert 'selected_route = state["commit_route"]' in launch
    assert "_fr13_fixed32_conv_flat_zeroelide_col0_kernel[grid]" in kernel_launch
    assert "_fr13_fixed32_conv_direct_col0_kernel[grid]" in kernel_launch
    assert "_fr13_resolve_fixed32_conv_flat_commit" not in launch


def test_channel_kernel_keeps_low_cta_map_and_elides_zero_source_loads() -> None:
    body = _function_text("_fr13_fixed32_conv_channel_zeroelide_col0_kernel")
    assert "offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)" in body
    assert "tl.static_range(0, LIVE_SOURCE_COLS)" in body
    assert "state_src + leaf_node * CONV_L + state_col" in body
    assert "tl.zeros((BLOCK_C,), dtype=tl.bfloat16)" in body
    assert "tl.static_range(LIVE_SOURCE_COLS, CONV_L)" in body
    zero_loop = body[body.index("tl.static_range(LIVE_SOURCE_COLS, CONV_L)") :]
    assert "tl.load(" not in zero_loop


def test_channel_route_is_preseeded_and_mutually_exclusive() -> None:
    preseed = _function_text("preseed_fixed32_conv_col0_pregather")
    launch = _function_text("launch_fixed32_conv_commit_to_col0")
    kernel_launch = _function_text(
        "_fr13_fixed32_conv_commit_kernel_launch"
    )
    assert "channel_commit = _fr13_resolve_fixed32_conv_channel_commit()" in preseed
    assert "zero-eliding candidates are mutually exclusive" in preseed
    assert '"commit_channel_zeroelide": channel_commit' in preseed
    assert "elif channel_commit:" in preseed
    assert (
        "_fr13_fixed32_conv_channel_zeroelide_col0_kernel[\n"
        "                    channel_grid\n"
        "                ](" in preseed
    )
    assert 'if state["commit_channel_zeroelide"]:' in launch
    assert "_fr13_fixed32_conv_channel_byte_gate(" in launch
    assert (
        "_fr13_fixed32_conv_channel_zeroelide_col0_kernel[grid]"
        in kernel_launch
    )
    assert "_fr13_resolve_fixed32_conv_channel_commit" not in launch


def test_channel_gate_is_real_event_only_exact_and_reference_served() -> None:
    gate = _function_text("_fr13_fixed32_conv_channel_byte_gate")
    marker = gate.index("_fr13_fixed32_conv_channel_real_event_marker()")
    accepted_lens_read = gate.index("accepted_lens[:batch].tolist()")

    assert marker < accepted_lens_read
    assert 'saved_conv_rows = tuple(' in gate
    assert 'saved_ssm_rows = tuple(' in gate
    assert gate.count("_fr13_fixed32_conv_commit_kernel_launch(") == 2
    assert "route=_FR13_FIXED32_CONV_COMMIT_ROUTE" in gate
    assert "route=_FR13_FIXED32_CONV_CHANNEL_COMMIT_ROUTE" in gate
    assert gate.count("_fr13_fixed32_tensor_bits_equal(") == 3
    assert '"commit_channel_byte_gate_coverage_mask_by_batch"' in gate
    assert 'coverage_mask = int(coverage_by_batch[batch])' in gate
    assert 'attempts_by_batch[batch] += 1' in gate
    assert 'passed_by_batch[batch] = coverage_mask == full_mask' in gate
    assert "finally:" in gate
    assert gate.rstrip().endswith("return False")


def test_channel_gate_contract_binds_all_depths_and_collateral_rows() -> None:
    preseed = _function_text("preseed_fixed32_conv_col0_pregather")

    assert '"commit_channel_byte_gate": (' in preseed
    assert '"real_swe_all_reachable_accepted_lengths_0_11"' in preseed
    assert '"commit_channel_byte_gate_raw_compare": (' in preseed
    assert '"torch_equal_uint8" if channel_commit else None' in preseed
    assert '"commit_channel_byte_gate_collateral": (' in preseed
    assert '"companion_ssm_running_rows" if channel_commit else None' in preseed
    assert '"commit_channel_unseen_length_route": (' in preseed
    assert '"shadow_then_reference" if channel_commit else None' in preseed


def test_flat_contract_is_bound_to_deployed_geometry() -> None:
    assert _constant("_FR13_FIXED32_CONV_FLAT_C") == 10_240
    assert _constant("_FR13_FIXED32_CONV_FLAT_L") == 34
    assert _constant("_FR13_FIXED32_CONV_FLAT_SOURCE_ROWS") == 36
    assert _constant("_FR13_FIXED32_CONV_FLAT_LIVE_SOURCE_COLS") == 3
    assert (
        _constant("_FR13_FIXED32_CONV_FLAT_COMMIT_ROUTE")
        == "fixed32_flat_zeroelide_source_col0"
    )
    assert (
        _constant("_FR13_FIXED32_CONV_CHANNEL_COMMIT_ROUTE")
        == "fixed32_channel_zeroelide_source_col0"
    )
