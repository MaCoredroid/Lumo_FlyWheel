import ast
import os
import stat
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
    assert "for i_t in tl.range(0, T):" in kernel
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
    assert "accepted_paths + i_n * PATH_CAP + path_offset" in kernel
    assert "accepted = tl.load(accepted_lens + i_n).to(tl.int64)" in kernel
    assert "T = tl.minimum(tl.maximum(accepted, 0) + 1, PATH_CAP)" in kernel
    assert 'state["accepted_lens"]' in launch
    assert 'state["cu"]' not in launch
    assert "PATH_CAP=16" in launch


def test_layer_batch_publishes_only_the_final_running_state() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    loop = kernel[kernel.index("for i_t in tl.range(0, T):") :]

    assert "final_state_idx" not in kernel
    assert "p_ht" not in kernel
    assert loop.count("tl.store(") == 1
    assert "tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)" in kernel


def test_layer_batch_hoists_constant_gate_coefficients() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    loop_at = kernel.index("for i_t in tl.range(0, T):")

    assert kernel.index("b_a_scale = -tl.exp(") < loop_at
    assert kernel.index("b_dt_bias = tl.load(p_dt_bias)") < loop_at
    loop = kernel[loop_at:]
    assert "tl.load(p_a_log)" not in loop
    assert "tl.load(p_dt_bias)" not in loop
    assert "b_g = b_a_scale * softplus_x" in loop


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


def test_layer_batch_candidate_loads_live_ring_rows_without_staging() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    launch = _text("_fr13_fixed32_committer_native_layer_batch")
    body = _text("_fr13_fixed32_committer_graph_body")
    preseed = _text("preseed_fixed32_committer_graph")

    candidate_start = body.index("if use_layer_batch:")
    candidate = body[candidate_start : body.index("else:", candidate_start)]
    assert "k_rings" in kernel
    assert "v_rings" in kernel
    assert "a_rings" in kernel
    assert "b_rings" in kernel
    assert "accepted_paths" in kernel
    assert "path_node = tl.load(" in kernel
    assert "node * RING_K_N_STRIDE" in kernel
    assert "node * RING_V_N_STRIDE" in kernel
    assert 'state["kbuf"]' not in launch
    assert 'state["vbuf"]' not in launch
    assert "k_selected" not in candidate
    assert "torch.where(" not in candidate
    assert '"neutralizations": 0' in preseed
    assert '"ring_gathers": 0' in preseed
    assert '"direct_ring_loads": True' in preseed
    assert '"direct_ring_inputs": 4' in preseed
    assert '"candidate_staging_launches": 0' in preseed
    assert '"gate_coefficients_hoisted": True' in preseed


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
    assert '"layer_batch_byte_gate_coverage_mask": (' in preseed
    assert '"fused_calls": 48' in preseed
    assert '"fused_calls": 1' in preseed
    assert '"state_only_output_elided": True' in preseed
    assert '"active_length_recurrence": True' in preseed
    assert '"final_state_store_once": True' in preseed


def test_accepted_length_mask_covers_zero_through_eleven() -> None:
    node = _function("_fr13_fixed32_committer_accepted_length_mask")
    namespace: dict[str, object] = {
        "_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH": 11,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL_PATH), "exec"),
        namespace,
    )
    length_mask = namespace[node.name]

    assert length_mask((0,), batch=1) == 0x0001
    assert length_mask((11,), batch=1) == 0x0800
    assert length_mask((0, 7, 11, 7), batch=4) == 0x0881
    with pytest.raises(RuntimeError, match="qualification input drift"):
        length_mask((0,), batch=4)
    with pytest.raises(RuntimeError, match="qualification input drift"):
        length_mask((12,), batch=1)


def test_real_event_arm_requires_private_authenticated_swe_file(tmp_path: Path) -> None:
    node = _function("_fr13_fixed32_committer_layer_batch_real_event_marker")
    namespace = {
        "os": os,
        "stat": stat,
        "_FR13_FIXED32_COMMITTER_LAYER_BATCH_REAL_EVENT": str(
            tmp_path / "default.arm"
        ),
        "_FR13_FIXED32_COMMITTER_TASK_ID_CHARACTERS": frozenset(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/"
        ),
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(KERNEL_PATH), "exec"),
        namespace,
    )
    marker = namespace[node.name]
    path = tmp_path / "qualification.arm"

    assert marker(str(path)) is None
    path.write_text("swe_verified:astropy__astropy-12907\n", encoding="ascii")
    path.chmod(0o400)
    assert marker(str(path)) == "swe_verified:astropy__astropy-12907"
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="private read-only regular file"):
        marker(str(path))


def test_byte_gate_covers_only_authenticated_real_event_depths() -> None:
    gate = _text("_fr13_fixed32_committer_layer_batch_byte_gate")
    replay = _text("_fr13_fixed32_committer_replay")
    bit_compare = _text("_fr13_fixed32_tensor_bits_equal")

    authenticated = (
        "if _fr13_fixed32_committer_layer_batch_real_event_marker() is None:"
    )
    length_mask = "_fr13_fixed32_committer_accepted_length_mask("
    reference = "reference_graph.replay()"
    candidate = "candidate_graph.replay()"
    compare = "_fr13_fixed32_tensor_bits_equal("
    assert authenticated in gate
    assert gate.index(authenticated) < gate.index(length_mask)
    assert gate.index(length_mask) < gate.index(reference) < gate.index(candidate)
    assert compare in gate
    assert "torch.equal(" in bit_compare
    assert bit_compare.count(".view(torch.uint8)") == 2
    assert 'coverage_mask |= event_mask' in gate
    assert 'state["layer_batch_byte_gate_coverage_mask"] = coverage_mask' in gate
    assert "coverage_mask == full_mask" in gate
    assert gate.rstrip().endswith("return False")
    assert "finally:\n        restore()" in gate
    assert "graph = state[\"reference_graph\"]" in replay
    assert replay.index("_fr13_fixed32_committer_layer_batch_byte_gate(") < replay.index(
        "graph.replay()"
    )


def test_byte_gate_serves_reference_once_then_allows_covered_depth() -> None:
    nodes = [
        _function("_fr13_fixed32_committer_accepted_length_mask"),
        _function("_fr13_fixed32_committer_layer_batch_byte_gate"),
    ]
    replay_order: list[str] = []

    class _Rows:
        def clone(self):
            return self

    class _Bank:
        def index_select(self, _dimension, _rows):
            return _Rows()

        def index_copy_(self, _dimension, _rows, _saved):
            return None

    class _Graph:
        def __init__(self, label: str) -> None:
            self.label = label

        def replay(self) -> None:
            replay_order.append(self.label)

    class _Lens:
        device = "cuda:0"

        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return list(self.values)

    class _Cuda:
        @staticmethod
        def synchronize(_device) -> None:
            return None

    namespace = {
        "_FR13_FIXED32_COMMITTER_MAX_ACCEPTED_LENGTH": 11,
        "_FR13_FIXED32_COMMITTER_ACCEPTED_LENGTH_FULL_MASK": 0x0FFF,
        "_fr13_fixed32_committer_layer_batch_real_event_marker": (
            lambda: "swe_verified:astropy__astropy-12907"
        ),
        "_fr13_fixed32_validate_running_rows": (
            lambda **_kwargs: [object() for _ in range(48)]
        ),
        "_fr13_fixed32_tensor_bits_equal": lambda _left, _right: True,
        "torch": type("_Torch", (), {"cuda": _Cuda}),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(KERNEL_PATH), "exec"),
        namespace,
    )
    gate = namespace["_fr13_fixed32_committer_layer_batch_byte_gate"]
    state = {
        "batch": 1,
        "bank_rows": 8,
        "layer_batch": True,
        "layer_batch_byte_gate_passed": False,
        "layer_batch_byte_gate_attempts": 0,
        "layer_batch_byte_gate_coverage_mask": 0,
        "accepted_lens": _Lens([0]),
        "reference_graph": _Graph("reference"),
        "graph": _Graph("candidate"),
    }
    banks = [_Bank() for _ in range(48)]

    assert gate(state=state, banks_list=banks, spec_state_indices=object()) is False
    assert replay_order == ["reference", "candidate"]
    assert state["layer_batch_byte_gate_coverage_mask"] == 0x0001
    assert state["layer_batch_byte_gate_attempts"] == 1
    assert state["layer_batch_byte_gate_passed"] is False

    assert gate(state=state, banks_list=banks, spec_state_indices=object()) is True
    assert replay_order == ["reference", "candidate"]
    assert state["layer_batch_byte_gate_attempts"] == 1

    state["layer_batch_byte_gate_coverage_mask"] = 0x07FF
    state["accepted_lens"].values = [11]
    assert gate(state=state, banks_list=banks, spec_state_indices=object()) is False
    assert state["layer_batch_byte_gate_coverage_mask"] == 0x0FFF
    assert state["layer_batch_byte_gate_passed"] is True


def test_counters_expose_per_batch_gate_state_for_timing_boundaries() -> None:
    counters = _text("fixed32_committer_counters")

    assert 'fast_route.get("states_by_batch", {})' in counters
    assert '"layer_batch_gate_passed_by_batch"' in counters
    assert '"layer_batch_gate_attempts_by_batch"' in counters
    assert '"layer_batch_gate_coverage_mask_by_batch"' in counters
    assert 'state.get("layer_batch_byte_gate_passed", False)' in counters
    assert 'state.get("layer_batch_byte_gate_attempts", -1)' in counters
    assert 'state.get("layer_batch_byte_gate_coverage_mask", -1)' in counters


def test_layer_programs_have_disjoint_layer_state_and_shared_read_only_paths() -> None:
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")

    assert "i_l = i_lnh // layer_span" in kernel
    assert "state_bank = bank_anchor +" in kernel
    assert "spec_state_indices + i_l * SPEC_L_STRIDE" in kernel
    assert "i_l * RING_K_L_STRIDE" in kernel
    assert "i_l * RING_V_L_STRIDE" in kernel
    assert "accepted_paths" in kernel
    assert "accepted_lens" in kernel


def test_candidate_binds_physical_alias_row_uniqueness_guard() -> None:
    preseed = _text("preseed_fixed32_committer_graph")
    guard = _text("validate_fixed32_conv_commit_rows")
    conv_commit = _text("launch_fixed32_conv_commit_to_col0")

    assert '"physical_alias_row_uniqueness_guard": (' in preseed
    assert '"validate_fixed32_conv_commit_rows"' in preseed
    assert "bank_alias_ids.view(48, 1) * bank_rows + running_rows" in guard
    assert "distinct_destinations" in guard
    assert conv_commit.index("validate_fixed32_conv_commit_rows(") < conv_commit.index(
        "_fr13_fixed32_conv_direct_col0_kernel[grid]("
    )
    kernel = _text("_fr13_fixed32_committer_native_layer_batch_kernel")
    assert "disjoint physical ``(alias," in kernel
    assert "row)`` destinations" in kernel


def test_launcher_materializes_worker_visible_arm_only_when_requested() -> None:
    launcher = LAUNCHER_PATH.read_text()
    server_launcher = (
        ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
    ).read_text()

    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH=${" in launcher
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH must be exactly 0 or 1" in launcher
    assert 'if [[ "$FR13_FIXED32_COMMITTER_LAYER_BATCH" == "1" ]]' in launcher
    assert '"$ARMDIR/logs/fr13_fixed32_committer_layer_batch.arm"' in launcher
    assert "committer layer-batch arm requires a fixed32 kind" in launcher
    assert "FR13_FIXED32_COMMITTER_LAYER_BATCH_QUALIFICATION=${" in launcher
    assert "CFWD layer-batch qualification is fixed32 B1/sequential only" in launcher
    assert "--fixed32-committer-layer-batch-real-event-arm" in launcher
    assert "FIXED32_COMMITTER_LAYER_BATCH_REAL_EVENT_ARM_PATH" in launcher
    assert (
        'rm -f -- "$FIXED32_COMMITTER_LAYER_BATCH_REAL_EVENT_ARM_PATH"'
        in launcher
    )
    assert "fr13_fixed32_committer_layer_batch.real_event.arm" in server_launcher


def test_observer_preserves_logical_layers_and_candidate_physical_calls() -> None:
    patcher = PATCHER_PATH.read_text()

    assert 'layer_batch = committer_contract.get("layer_batch", False)' in patcher
    assert "expected_fused_calls = 1 if layer_batch is True else 48" in patcher
    assert 'committer_contract.get("state_only_output_elided") is not True' in patcher
    assert 'committer_contract.get("active_length_recurrence") is not True' in patcher
    assert 'committer_contract.get("final_state_store_once") is not True' in patcher
    assert 'committer_contract.get("direct_ring_loads") is not True' in patcher
    assert 'committer_contract.get("direct_ring_inputs", -1)' in patcher
    assert 'committer_contract.get("candidate_staging_launches", -1)' in patcher
    assert 'committer_contract.get("gate_coefficients_hoisted") is not True' in patcher
    assert '"physical_alias_row_uniqueness_guard"' in patcher
    assert '!= "validate_fixed32_conv_commit_rows"' in patcher
    assert '!= "real_swe_all_reachable_accepted_lengths_0_11"' in patcher
    assert 'committer_contract.get("accepted_length_max", -1)' in patcher
    assert ") != 0x0FFF" in patcher
    assert '!= "torch_equal_uint8"' in patcher
    assert '!= "shadow_then_reference"' in patcher
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
    event["committer"]["ring_gather_ops"] = 0
    event["committer"]["ring_layer_path_rows"] = 0
    validated = census.validate_event(event, source="layer-batch-candidate")

    assert validated.normalized_work["committer"]["layers"] == 48
    assert (
        validated.normalized_work["committer"]["fused_layer_calls_per_event"]
        == 1
    )
    assert validated.normalized_work["committer"]["ring_gather_ops"] == 0
    assert (
        validated.normalized_work["committer"][
            "ring_layer_path_rows_per_request"
        ]
        == 0
    )

    event["committer"]["fused_layer_calls"] = 2
    with pytest.raises(census.CensusError, match="expected 1 or 48"):
        census.validate_event(event, source="invalid-layer-batch-count")
