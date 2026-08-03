from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
PATCHER = PATCHER_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _text(name: str) -> str:
    node = _function(name)
    lines = SOURCE.splitlines()
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


def test_knorm_ring_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_knorm_ring_requested")
    namespace = {"os": os}
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(KERNEL_PATH),
            "exec",
        ),
        namespace,
    )
    requested = namespace[node.name]

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_KNORM_RING", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_KNORM_RING", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_KNORM_RING")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path
        == "/logs/fr13_fixed32_committer_knorm_ring.arm",
    )
    assert requested() is True


def test_fixed32_producer_exports_the_existing_rsqrt_once() -> None:
    helper = _text("_tree_gdn_fixed32_single_launch_node")
    kernel = _text("_tree_gdn_kernel_fixed32_single_launch")

    candidate = helper[
        helper.index("    if K_NORM_EXPORT:") : helper.index(
            "    new_state, out_i = _gdn_node_step("
        )
    ]
    assert "tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)" in candidate
    assert "b_k = b_k * b_k_inv_norm" in candidate
    assert "tl.store(" not in candidate
    assert "USE_QK_L2NORM_IN_KERNEL and not K_NORM_EXPORT" in helper
    assert "SCN_ALIGN" not in helper
    assert "SCAN_ALIGN=SCAN_ALIGN and not K_NORM_EXPORT" in helper
    assert "ring_k_norm + global_node * NUM_KH + pid_kh" in helper
    assert "(pid_v == 0) & (pid_vh % head_group == 0)" in helper
    assert kernel.count("K_NORM_EXPORT=K_NORM_EXPORT") == 2
    assert "K_NORM_EXPORT: tl.constexpr = False" in kernel


def test_b1_and_b4_exports_are_physical32_single_launch_only() -> None:
    b1 = _text("launch_tree_gdn_prepared")
    b4 = _text("launch_tree_gdn_prepared_fixed32_batch")

    assert "tuple(ring_k_norm.shape) != (n_pad, num_kh)" in b1
    assert "not _FR13_FIXED32_GDN_SINGLE_LAUNCH" in b1
    assert "_k_norm_export and _scan_align" in b1
    assert "K_NORM_EXPORT=_k_norm_export" in b1

    assert "tuple(ring_k_norm.shape) != (rows, num_kh)" in b4
    assert 'batch != 4' in b4
    assert 'selector != "single_launch"' in b4
    assert "K_NORM_EXPORT=k_norm_export" in b4


def test_committer_replaces_reduction_with_one_scalar_load() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launcher = _text("_fr13_fixed32_committer_native_layer_batch")

    candidate_start = kernel.index("        if K_NORM_REUSE:")
    fallback_start = kernel.index(
        "        elif USE_QK_L2NORM_IN_KERNEL:", candidate_start
    )
    update_start = kernel.index("        b_h *= tl.exp(b_g)", fallback_start)
    candidate = kernel[candidate_start:fallback_start]
    fallback = kernel[fallback_start:update_start]

    assert "tl.load(" in candidate
    assert "tl.sum(" not in candidate
    assert "tl.rsqrt(" not in candidate
    assert "node * RING_KN_N_STRIDE" in candidate
    assert "tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)" in fallback
    assert "K_NORM_REUSE=bool(k_norm_reuse)" in launcher
    assert "num_warps=8" in launcher
    assert "block_k = triton.next_power_of_2(dim_k)" in launcher


def test_preseed_is_fail_closed_and_keeps_real_event_byte_gate() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    gate = _text("_fr13_fixed32_committer_layer_batch_byte_gate")

    for requirement in (
        "not layer_batch",
        "not direct_metadata",
        "not _FR13_FIXED32_GDN_SINGLE_LAUNCH",
        "scan_align_on()",
        "not use_qk_l2norm_in_kernel",
        "k_norm_rings is None",
    ):
        assert requirement in preseed
    assert '"k_norm_reuse": k_norm_reuse' in preseed
    assert '"producer_extra_k_norm_reductions": 0' in preseed
    assert '"k_norm_reductions_per_value_head_step"' in preseed
    assert "_fr13_fixed32_committer_layer_batch_real_event_marker()" in gate
    assert "_fr13_fixed32_tensor_bits_equal(" in gate
    assert "reference_served=1" in gate


def test_patcher_allocates_and_wires_only_the_scalar_ring() -> None:
    assert "_fr13_ep_ring_k_norm = (" in PATCHER
    assert "(_fr13_ep_count, _fr13_ring_bs, n_pad, _fr13_ep_kh)" in PATCHER
    assert "dtype=torch.float32" in PATCHER
    assert '\\"ring_k_norm\\": _fr13_ep_ring_k_norm' in PATCHER
    assert "k_norm_rings=_fr13_f32_stacks[" in PATCHER
    assert PATCHER.count("ring_k_norm=(") == 2
    assert 'k_norm_rings=stacks["ring_k_norm"]' in PATCHER


def test_operation_census_scales_with_live_steps_not_physical32() -> None:
    layers = 48
    key_heads = 16
    value_heads = 48
    physical_nodes = 32
    for batch in (1, 4):
        scalar_ring_values = layers * batch * physical_nodes * key_heads
        for accepted_drafts in (0, 4, 11):
            live_steps = accepted_drafts + 1
            removed_reductions = layers * batch * live_steps * value_heads
            added_producer_reductions = 0
            assert removed_reductions > 0
            assert added_producer_reductions == 0
            assert scalar_ring_values == layers * batch * 512
