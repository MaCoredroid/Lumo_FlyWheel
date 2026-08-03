from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fr13_fixed32_topology import PHYSICAL_PARENT  # noqa: E402
from lumo_flywheel_serving.fr13_tree_conv_fused import (  # noqa: E402
    build_tree_conv_state_src_indices,
)


KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
VARIANT_PATH = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
LAUNCHER_PATH = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
RUNNER_PATH = ROOT / "scripts/fr13_run_treeconv_zero_tail_live_gate.sh"
SWE_RUNNER_PATH = ROOT / "scripts/run_swe_bench_q36_a.py"


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1
    return matches[0]


@pytest.mark.parametrize("batch", (1, 4))
def test_physical32_zero_tail_is_byte_exact_for_every_leaf(batch: int) -> None:
    channels = 37
    state_len = 34
    source_rows = 36
    live_cols = 3
    state_src = build_tree_conv_state_src_indices(
        parent=list(PHYSICAL_PARENT),
        width=4,
        state_len=state_len,
        device="cpu",
    ).view(32, state_len)
    assert torch.all(state_src[:, live_cols:] == source_rows - 1)

    generator = torch.Generator().manual_seed(3200 + batch)
    sources = torch.randn(
        (batch, source_rows, channels),
        generator=generator,
        dtype=torch.bfloat16,
    )
    sources[:, source_rows - 1].zero_()
    for request in range(batch):
        for leaf in range(32):
            incumbent = sources[request].index_select(
                0, state_src[leaf]
            ).transpose(0, 1).contiguous()
            candidate = torch.zeros_like(incumbent)
            candidate[:, :live_cols] = sources[request].index_select(
                0, state_src[leaf, :live_cols]
            ).transpose(0, 1)
            assert torch.equal(
                incumbent.view(torch.int16), candidate.view(torch.int16)
            )


def test_candidate_is_default_off_and_exact_geometry_fail_closed() -> None:
    source = KERNEL_PATH.read_text()
    tree = ast.parse(source)
    selector = ast.unparse(
        _function(tree, "_fr13_fixed32_conv_commit_zero_tail_requested")
    )
    preseed = ast.unparse(_function(tree, "preseed_fixed32_conv_col0_pregather"))

    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL', '0'" in selector
    assert "raw not in ('0', '1')" in selector
    assert "fr13_fixed32_conv_commit_zero_tail.arm" in source
    for needle in (
        "anchor.dtype != torch.bfloat16",
        "conv_c != 10240",
        "conv_l != 34",
        "source_rows != 36",
        "value != source_rows - 1",
    ):
        assert needle in preseed


def test_both_direct_kernels_keep_incumbent_and_zero_tail_specializations() -> None:
    tree = ast.parse(KERNEL_PATH.read_text())
    for name in (
        "_fr13_fixed32_conv_direct_col0_kernel",
        "_fr13_fixed32_conv_direct_col0_metadata_kernel",
    ):
        kernel = ast.unparse(_function(tree, name))
        assert "ZERO_TAIL: tl.constexpr" in kernel
        assert "LIVE_STATE_COLS: tl.constexpr" in kernel
        assert "if ZERO_TAIL and state_col >= LIVE_STATE_COLS:" in kernel
        assert "tl.zeros((BLOCK_C,), dtype=tl.bfloat16)" in kernel
        assert "state_src + leaf_node * CONV_L + state_col" in kernel
        assert "for state_col in tl.static_range(0, CONV_L)" in kernel


def test_selector_is_forwarded_and_materialized_worker_visible() -> None:
    variant = VARIANT_PATH.read_text()
    launcher = LAUNCHER_PATH.read_text()

    assert (
        "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=${"
        "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL:-0}"
    ) in variant
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL must be exactly 0 or 1" in variant
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL" in variant
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL must be exactly 0 or 1" in launcher
    assert 'fr13_fixed32_conv_commit_zero_tail.arm"' in launcher
    assert (
        '-e FR13_FIXED32_CONV_COMMIT_ZERO_TAIL="${'
        'FR13_FIXED32_CONV_COMMIT_ZERO_TAIL:-0}"'
    ) in launcher
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB" in variant
    assert (
        '-e FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB="${'
        'FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB:-0}"'
    ) in launcher


def test_real_task_gate_is_default_off_stock_serving_and_bounded() -> None:
    kernel = KERNEL_PATH.read_text()
    variant = VARIANT_PATH.read_text()
    runner = RUNNER_PATH.read_text()
    swe_runner = SWE_RUNNER_PATH.read_text()

    assert '"FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB", "0"' in kernel
    assert "_launch_direct(zero_tail=True)" in kernel
    assert "_launch_direct(zero_tail=False)" in kernel
    assert '"reference_restored_and_served": True' in kernel
    assert '"timing_eligible": False' in kernel
    assert "not 1 <= limit <= 320" in kernel
    assert "_fr13_fixed32_conv_zero_tail_compare_kernel[grid]" in kernel
    assert "treeconv_zero_tail_count_enable" in kernel
    assert "fixed32_conv_zero_tail_live_prepare_replay" in kernel
    assert "fixed32_conv_zero_tail_live_finalize" in kernel
    assert "eager-only" not in ast.unparse(
        _function(ast.parse(kernel), "launch_fixed32_conv_commit_to_col0")
    )
    assert "tree-conv zero-tail byte diagnostic must be the only" in variant
    assert "treeconv_zero_tail_graph_diagnostic" in swe_runner
    assert "ENFORCE_EAGER=0" in runner
    assert '--final-flush "$ARMDIR/fixed32_final_flush.json"' in runner
    assert "--proxy-ledger" in runner
    assert "--task-root" in runner
    assert "config/fr13_fixed32/subset_b4_four.json" in runner
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in runner
    assert "AGENT_WALL_S=5400" in runner
    assert "FR13_DRAFT_VOCAB_K=65536" in runner
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL=0" in runner
    assert "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB=1" in runner
    assert "PROBE_ONLY" not in runner
    assert "ACCEPT_SPEED_PROBE" not in runner
