from __future__ import annotations

import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
SOURCE = KERNEL_PATH.read_text(encoding="utf-8")
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


def test_decay_ring_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_decay_ring_requested")
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), KERNEL_PATH, "exec"),
        namespace,
    )
    requested = namespace[node.name]

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_DECAY_RING", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_DECAY_RING", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_DECAY_RING")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/logs/fr13_fixed32_committer_decay_ring.arm",
    )
    assert requested() is True


def test_producer_reuses_its_exact_decay_and_elides_dead_raw_gate_stores() -> None:
    node = _text("_tree_gdn_fixed32_single_launch_node")
    helper = _text("_gdn_node_step_precomputed_decay")
    kernel = _text("_tree_gdn_kernel_fixed32_single_launch")

    assert node.count("b_decay = tl.exp(b_g)") == 1
    assert "b_decay if DECAY_EXPORT else b_g" in node
    assert "if RING_EXPORT and not DECAY_EXPORT:" in node
    assert "if DECAY_EXPORT:" in node
    assert "b_decay," in node
    assert "state_i *= b_decay" in helper
    assert "tl.exp(" not in helper
    assert "DECAY_EXPORT: tl.constexpr = False" in kernel
    assert kernel.count("DECAY_EXPORT=DECAY_EXPORT") == 2


def test_b1_and_b4_decay_exports_require_the_gate_ring_route() -> None:
    b1 = _text("launch_tree_gdn_prepared")
    b4 = _text("launch_tree_gdn_prepared_fixed32_batch")

    assert "_decay_export and not _gate_export" in b1
    assert "DECAY_EXPORT=_decay_export" in b1
    assert "decay_export and not gate_export" in b4
    assert "DECAY_EXPORT=decay_export" in b4


def test_committer_uses_exported_decay_without_an_exponential() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launcher = _text("_fr13_fixed32_committer_native_layer_batch")
    start = kernel.index("        if DECAY_REUSE:")
    end = kernel.index("        b_v -=", start)
    decay_update = kernel[start:end]

    assert "b_h *= b_g_or_decay" in decay_update
    assert "b_h *= tl.exp(b_g_or_decay)" in decay_update
    assert decay_update.count("tl.exp(") == 1
    assert "DECAY_REUSE=bool(decay_reuse)" in launcher
    assert "decay_reuse and not gate_reuse" in launcher
    assert '{"maxnreg": 169} if decay_reuse else {}' in launcher


def test_decay_preseed_is_cumulative_fail_closed_and_byte_gated() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    graph_body = _text("_fr13_fixed32_committer_graph_body")
    replay = _text("_fr13_fixed32_committer_replay")

    for requirement in (
        "not gate_reuse",
        "not k_norm_reuse",
        "not layer_batch",
        "not direct_metadata",
        "not _FR13_FIXED32_GDN_SINGLE_LAUNCH",
        "scan_align_on()",
        "not use_qk_l2norm_in_kernel",
        "gate_rings is None",
    ):
        assert requirement in preseed
    assert '"decay_reuse": decay_reuse' in preseed
    assert '"raw_ab_ring_store_elision": decay_reuse' in preseed
    assert '"committer_decay_exponentials_per_value_head_step"' in preseed
    assert "_fr13_fixed32_committer_layer_batch_byte_gate(" in replay
    assert "decay_reuse=decay_reuse" in graph_body


def test_decay_operation_census_is_physical32_for_b1_and_b4() -> None:
    layers = 48
    physical_nodes = 32
    value_heads = 48
    raw_gate_values = 2
    bf16_bytes = 2

    assert (
        layers * 1 * physical_nodes * value_heads * raw_gate_values * bf16_bytes
        == 294_912
    )
    assert (
        layers * 4 * physical_nodes * value_heads * raw_gate_values * bf16_bytes
        == 1_179_648
    )
    for batch in (1, 4):
        for accepted_drafts in (0, 4, 11):
            live_steps = accepted_drafts + 1
            removed_exponentials = layers * batch * live_steps * value_heads
            assert removed_exponentials > 0
