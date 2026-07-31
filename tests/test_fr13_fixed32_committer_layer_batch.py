import ast
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
SOURCE = KERNEL_PATH.read_text()
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


def test_layer_batch_arm_is_explicit_and_default_off(monkeypatch) -> None:
    node = _function("_fr13_fixed32_committer_layer_batch_requested")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL_PATH), "exec"), namespace)
    requested = namespace[node.name]

    monkeypatch.delenv("FR13_FIXED32_COMMITTER_LAYER_BATCH", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: False)
    assert requested() is False

    monkeypatch.setenv("FR13_FIXED32_COMMITTER_LAYER_BATCH", "1")
    assert requested() is True
    monkeypatch.delenv("FR13_FIXED32_COMMITTER_LAYER_BATCH")
    monkeypatch.setattr(
        os.path,
        "exists",
        lambda path: path == "/logs/fr13_fixed32_committer_layer_batch.arm",
    )
    assert requested() is True


def test_layer_batch_kernel_keeps_native_recurrence_and_geometry() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launch = _text("_fr13_fixed32_committer_native_layer_batch")

    assert '@triton.jit(do_not_specialize=["N", "T"])' in SOURCE
    assert "for i_t in range(0, T):" in kernel
    assert "b_h *= tl.exp(b_g)" in kernel
    assert "b_v -= tl.sum(b_h * b_k[None, :], 1)" in kernel
    assert "b_v *= b_beta" in kernel
    assert "b_h += b_v[:, None] * b_k[None, :]" in kernel
    assert "state_bank = bank_anchor + tl.load(bank_off16 + i_l) * 4" in kernel
    assert "_gdn_node_step" not in kernel
    assert "block_v = min(triton.next_power_of_2(dim_v), 32)" in launch
    assert "num_warps=4" in launch
    assert "num_stages=3" in launch
    assert "layers * batch * num_vh" in launch


def test_graph_keeps_native_reference_and_candidate_as_separate_captures() -> None:
    body = _text("_fr13_fixed32_committer_graph_body")
    preseed = _text("preseed_fixed32_committer_graph")

    assert "if use_layer_batch:" in body
    assert "_fr13_fixed32_committer_native_layer_batch(" in body
    assert "for layer in range(layers):" in body
    assert "fused_sigmoid_gating_delta_rule_update as _sg" in body
    assert "reference_graph = capture_graph(use_layer_batch=False)" in preseed
    assert "graph = capture_graph(use_layer_batch=True)" in preseed
    assert '"layer_batch_byte_gate_passed": not layer_batch' in preseed
    assert '"fused_calls": 48' in preseed
    assert '"fused_calls": 1' in preseed


def test_byte_gate_requires_real_nonzero_path_and_exact_state_bytes() -> None:
    gate = _text("_fr13_fixed32_committer_layer_batch_byte_gate")
    replay = _text("_fr13_fixed32_committer_replay")

    powered = 'if not bool((state["accepted_lens"] > 0).any().item()):'
    reference = "reference_graph.replay()"
    candidate = "candidate_graph.replay()"
    compare = "_fr13_fixed32_tensor_bits_equal("
    assert powered in gate
    assert gate.index(powered) < gate.index(reference) < gate.index(candidate)
    assert compare in gate
    assert "finally:\n        restore()" in gate
    assert "graph = state[\"reference_graph\"]" in replay
    assert replay.index("_fr13_fixed32_committer_layer_batch_byte_gate(") < replay.index(
        "graph.replay()"
    )


def test_layer_programs_have_disjoint_layer_state_and_shared_read_only_paths() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")

    assert "i_l = i_lnh // layer_span" in kernel
    assert "state_bank = bank_anchor +" in kernel
    assert "ssi = ssm_state_indices + i_l * SSI_L_STRIDE" in kernel
    assert "i_l * K_L_STRIDE" in kernel
    assert "i_l * V_L_STRIDE" in kernel
    assert "accepted_paths" not in kernel
    assert "accepted_lens" not in kernel
