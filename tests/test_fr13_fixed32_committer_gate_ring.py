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


def test_gate_ring_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_gate_ring_requested")
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

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_GATE_RING", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_GATE_RING", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_GATE_RING")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/logs/fr13_fixed32_committer_gate_ring.arm",
    )
    assert requested() is True


def test_fixed32_producer_exports_existing_raw_gate_math_once() -> None:
    helper = _text("_tree_gdn_fixed32_single_launch_node")
    kernel = _text("_tree_gdn_kernel_fixed32_single_launch")
    start = helper.index("        if GATE_EXPORT:")
    end = helper.index("    b_k_inv_norm = 1.0", start)
    candidate = helper[start:end]

    assert "x = b_raw_a + b_dt_bias" in candidate
    assert "tl.log(1.0 + tl.exp(x))" in candidate
    assert "b_g = -tl.exp(b_a_log) * softplus_x" in candidate
    assert "b_b = tl.sigmoid(b_raw_b.to(tl.float32))" in candidate
    assert candidate.count("tl.store(") == 2
    assert "mask=n_ok & (pid_v == 0)" in candidate
    assert "RAW_GATING=RAW_GATING and not GATE_EXPORT" in helper
    assert kernel.count("GATE_EXPORT=GATE_EXPORT") == 2
    assert "GATE_EXPORT: tl.constexpr = False" in kernel


def test_b1_and_b4_gate_exports_require_cumulative_exact_route() -> None:
    b1 = _text("launch_tree_gdn_prepared")
    b4 = _text("launch_tree_gdn_prepared_fixed32_batch")

    for source, shape in (
        (b1, "(n_pad, num_vh, 2)"),
        (b4, "(rows, num_vh, 2)"),
    ):
        assert "not _fr13_fixed32_committer_gate_ring_requested()" in source
        assert shape in source
        assert "not k_norm_export" in source or "not _k_norm_export" in source
        assert "GATE_EXPORT=" in source
    assert "not raw_gating" in b1
    assert "scan_align_on()" in b1


def test_committer_gate_candidate_is_two_scalar_loads_without_raw_math() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launcher = _text("_fr13_fixed32_committer_native_layer_batch")
    start = kernel.index("        if GATE_REUSE:")
    fallback = kernel.index("        else:", start)
    candidate = kernel[start:fallback]

    assert candidate.count("tl.load(") == 2
    assert "p_live_gate" in candidate
    assert "tl.exp(" not in candidate
    assert "tl.log(" not in candidate
    assert "tl.sigmoid(" not in candidate
    assert "a_rings" not in candidate
    assert "b_rings" not in candidate
    assert "GATE_REUSE=bool(gate_reuse)" in launcher
    assert "gate_rings if gate_reuse else k_rings" in launcher


def test_gate_preseed_is_fail_closed_and_retains_real_byte_gate() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    byte_gate = _text("_fr13_fixed32_committer_layer_batch_byte_gate")

    for requirement in (
        "not k_norm_reuse",
        "not layer_batch",
        "not direct_metadata",
        "not _FR13_FIXED32_GDN_SINGLE_LAUNCH",
        "scan_align_on()",
        "not use_qk_l2norm_in_kernel",
        "gate_rings is None",
    ):
        assert requirement in preseed
    assert '"gate_reuse": gate_reuse' in preseed
    assert '"producer_extra_gate_nonlinear_evaluations": 0' in preseed
    assert '"gate_scalar_loads_per_value_head_step"' in preseed
    assert "_fr13_fixed32_committer_layer_batch_real_event_marker()" in byte_gate
    assert "_fr13_fixed32_tensor_bits_equal(" in byte_gate
    assert "reference_served=1" in byte_gate


def test_patcher_allocates_and_wires_one_packed_fp32_gate_ring() -> None:
    assert "_fr13_ep_ring_gate = (" in PATCHER
    assert "(_fr13_ep_count, _fr13_ring_bs, n_pad, _fr13_ep_vh, 2)" in PATCHER
    assert "_fr13_fixed32_committer_gate_ring_requested()" in PATCHER
    assert '\\"ring_gate\\": _fr13_ep_ring_gate' in PATCHER
    assert "gate_rings=_fr13_f32_stacks[" in PATCHER
    assert PATCHER.count("ring_gate=(") == 2
    assert 'gate_rings=stacks["ring_gate"]' in PATCHER


def test_gate_operation_census_scales_with_live_steps() -> None:
    layers = 48
    value_heads = 48
    physical_nodes = 32
    for batch in (1, 4):
        producer_values = layers * batch * physical_nodes * value_heads * 2
        producer_bytes = producer_values * 4
        assert producer_bytes == 589_824 * batch
        for accepted_drafts in (0, 4, 11):
            live_steps = accepted_drafts + 1
            removed_gate_sets = layers * batch * live_steps * value_heads
            scalar_loads = removed_gate_sets * 2
            assert removed_gate_sets > 0
            assert scalar_loads == removed_gate_sets * 2
