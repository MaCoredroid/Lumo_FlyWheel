import ast
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER_PATH = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts/fr13_bigdenom_swe_serve_variant.sh"
SOURCE = KERNEL_PATH.read_text()
TREE = ast.parse(SOURCE)
sys.path.insert(0, str(ROOT / "scripts"))
import fr13_fixed32_work_census as census  # noqa: E402


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

    assert "@triton.jit" in SOURCE
    assert "for i_t in range(0, T):" in kernel
    assert "b_h *= tl.exp(b_g)" in kernel
    assert "b_v -= tl.sum(b_h * b_k[None, :], 1)" in kernel
    assert "b_v *= b_beta" in kernel
    assert "b_h += b_v[:, None] * b_k[None, :]" in kernel
    assert "b_o =" not in kernel
    assert "p_q =" not in kernel
    assert "p_o =" not in kernel
    assert "tl.store(p_o" not in kernel
    assert "state_bank = bank_anchor + tl.load(bank_off16 + i_l) * 4" in kernel
    assert "_gdn_node_step" not in kernel
    assert "block_v = min(triton.next_power_of_2(dim_v), 32)" in launch
    assert "num_warps=4" in launch
    assert "num_stages=3" in launch
    assert "layers * batch * num_vh" in launch


def test_layer_batch_recurrence_stops_after_root_plus_accepted_drafts() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launch = _text("_fr13_fixed32_committer_native_layer_batch")

    assert "accepted_lens," in kernel
    assert "cu_seqlens," not in kernel
    assert "bos = i_n * PATH_CAP" in kernel
    assert "accepted = tl.load(accepted_lens + i_n).to(tl.int64)" in kernel
    assert "T = tl.minimum(tl.maximum(accepted, 0) + 1, PATH_CAP)" in kernel
    assert 'state["accepted_lens"]' in launch
    assert 'state["cu"]' not in launch
    assert "PATH_CAP=16" in launch


def test_layer_batch_publishes_only_the_final_running_state() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    loop = kernel[kernel.index("for i_t in range(0, T):") :]

    assert "final_state_idx" not in kernel
    assert "p_ht" not in kernel
    assert loop.count("tl.store(") == 1
    assert "tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)" in kernel


def test_layer_batch_candidate_drops_only_dead_operator_output() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launch = _text("_fr13_fixed32_committer_native_layer_batch")
    body = _text("_fr13_fixed32_committer_graph_body")
    preseed = _text("preseed_fixed32_committer_graph")

    assert "q," not in kernel
    assert "o," not in kernel
    assert "scale," not in kernel
    assert 'state["qbuf"]' not in launch
    assert 'state["obuf"]' not in launch
    assert '"obuf":' not in preseed
    assert "q=state[\"qbuf\"]" in body
    assert "\n            _sg(" in body
    assert "= _sg(" not in body


def test_layer_batch_candidate_writes_masked_gathers_directly() -> None:
    body = _text("_fr13_fixed32_committer_graph_body")
    preseed = _text("preseed_fixed32_committer_graph")

    candidate = body[body.index("if use_layer_batch:") :]
    assert "torch.where(mask4, k_selected, 0.0, out=k_destination)" in candidate
    assert "torch.where(mask4, v_selected, 0.0, out=v_destination)" in candidate
    assert "torch.where(mask3, a_selected, -1e4, out=a_destination)" in candidate
    assert "torch.where(mask3, b_selected, 0.0, out=b_destination)" in candidate
    assert '"neutralizations": 0' in preseed
    assert '"direct_masked_gather_writes": True' in preseed


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
    assert '"state_only_output_elided": True' in preseed
    assert '"active_length_recurrence": True' in preseed
    assert '"final_state_store_once": True' in preseed


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


def test_counters_expose_per_batch_gate_state_for_timing_boundaries() -> None:
    counters = _text("fixed32_committer_counters")

    assert 'fast_route.get("states_by_batch", {})' in counters
    assert '"layer_batch_gate_passed_by_batch"' in counters
    assert '"layer_batch_gate_attempts_by_batch"' in counters
    assert 'state.get("layer_batch_byte_gate_passed", False)' in counters
    assert 'state.get("layer_batch_byte_gate_attempts", -1)' in counters


def test_layer_programs_have_disjoint_layer_state_and_shared_read_only_paths() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")

    assert "i_l = i_lnh // layer_span" in kernel
    assert "state_bank = bank_anchor +" in kernel
    assert "ssi = ssm_state_indices + i_l * SSI_L_STRIDE" in kernel
    assert "i_l * K_L_STRIDE" in kernel
    assert "i_l * V_L_STRIDE" in kernel
    assert "accepted_paths" not in kernel
    assert "accepted_lens" in kernel


def test_launcher_materializes_worker_visible_arm_only_when_requested() -> None:
    launcher = LAUNCHER_PATH.read_text()

    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH=${" in launcher
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH must be exactly 0 or 1" in launcher
    assert 'if [[ "$FR13_FIXED32_COMMITTER_LAYER_BATCH" == "1" ]]' in launcher
    assert '"$ARMDIR/logs/fr13_fixed32_committer_layer_batch.arm"' in launcher
    assert "committer layer-batch arm requires a fixed32 kind" in launcher


def test_observer_preserves_logical_layers_and_candidate_physical_calls() -> None:
    patcher = PATCHER_PATH.read_text()

    assert 'layer_batch = committer_contract.get("layer_batch", False)' in patcher
    assert "expected_fused_calls = 1 if layer_batch is True else 48" in patcher
    assert 'committer_contract.get("state_only_output_elided") is not True' in patcher
    assert 'committer_contract.get("active_length_recurrence") is not True' in patcher
    assert 'committer_contract.get("final_state_store_once") is not True' in patcher
    assert 'committer_contract.get("direct_masked_gather_writes") is not True' in patcher
    assert "expected_neutralizations = 0 if layer_batch is True else 5" in patcher
    assert '"layers": int(layer_count)' in patcher
    assert "ring_gathers * int(layer_count) * path_cap * batch" in patcher
    assert '"fused_layer_calls": fused_calls' in patcher


def test_work_census_accepts_only_reference_or_layer_batch_launch_count() -> None:
    event = census.reference_event(
        census.HYDRA_MODE,
        1,
        "layer-batch-candidate",
    )
    event["committer"]["fused_layer_calls"] = 1
    event["committer"]["neutralize_ops"] = 0
    validated = census.validate_event(event, source="layer-batch-candidate")

    assert validated.normalized_work["committer"]["layers"] == 48
    assert (
        validated.normalized_work["committer"]["fused_layer_calls_per_event"]
        == 1
    )

    event["committer"]["fused_layer_calls"] = 2
    with pytest.raises(census.CensusError, match="expected 1 or 48"):
        census.validate_event(event, source="invalid-layer-batch-count")
