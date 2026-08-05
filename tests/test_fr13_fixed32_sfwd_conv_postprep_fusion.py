from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_conv_postprep_fusion.py"
)
KERNEL_PATH = (
    ROOT
    / "src"
    / "lumo_flywheel_serving"
    / "fr13_sfwd_conv_postprep_fusion_kernel.py"
)
GENERATOR_PATH = (
    ROOT / "scripts" / "fr13_generate_sfwd_conv_postprep_fusion_kernel.py"
)
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
ARTIFACT = ROOT / "results" / "fr13_fixed32_sfwd_conv_postprep_fusion_20260803"

sys.path.insert(0, str(ROOT / "src"))
try:
    import triton  # noqa: F401
except ModuleNotFoundError:
    triton_stub = types.ModuleType("triton")

    def _jit(function=None, **_kwargs):
        return (lambda decorated: decorated) if function is None else function

    triton_stub.jit = _jit
    triton_stub.cdiv = lambda left, right: (left + right - 1) // right
    language_stub = types.ModuleType("triton.language")
    triton_stub.language = language_stub
    sys.modules["triton"] = triton_stub
    sys.modules["triton.language"] = language_stub

from lumo_flywheel_serving import (  # noqa: E402
    fr13_sfwd_conv_postprep_fusion as candidate,
)
from lumo_flywheel_serving.fr13_sfwd_prior_reuse_descriptorless import (  # noqa: E402
    fixed32_descriptorless_sources,
)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


def _byte_gate_source_manifest(tmp_path: Path, commit: str) -> tuple[Path, str]:
    manifest = tmp_path / "source_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": candidate.SOURCE_MANIFEST_SCHEMA,
                "candidate": candidate.CANDIDATE,
                "source_commit": commit,
                "files": {
                    relative: {
                        "bytes": len((ROOT / relative).read_bytes()),
                        "sha256": hashlib.sha256(
                            (ROOT / relative).read_bytes()
                        ).hexdigest(),
                    }
                    for relative in candidate.SOURCE_FILES
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    manifest.chmod(0o400)
    return manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _gate_incumbent(
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gate_input = a.to(torch.float32) + dt_bias.to(torch.float32)
    softplus = torch.where(
        gate_input > 0,
        gate_input + torch.log(1.0 + torch.exp(-gate_input)),
        torch.log(1.0 + torch.exp(gate_input)),
    )
    softplus = torch.where(gate_input <= 20.0, softplus, gate_input)
    g = -torch.exp(A_log.to(torch.float32)) * softplus
    beta = torch.sigmoid(b.to(torch.float32))
    return g, beta


def _incumbent_reference(
    *,
    x: torch.Tensor,
    prior: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h: int,
    hv: int,
    k: int,
    v: int,
) -> dict[str, torch.Tensor]:
    batch, rows, channels = x.shape
    source_rows = fixed32_descriptorless_sources()
    conv_out = torch.empty_like(x)
    for request in range(batch):
        source = torch.cat(
            (prior[request], x[request]), dim=0
        )
        for node, historical in enumerate(source_rows):
            window_rows = (*historical, 3 + node)
            acc = bias.to(torch.float32).clone()
            for tap, source_row in enumerate(window_rows):
                product = (
                    source[source_row].to(torch.bfloat16)
                    * weights[:, tap].to(torch.bfloat16)
                ).to(torch.bfloat16).to(torch.float32)
                acc = acc + product
            conv_out[request, node] = (
                acc / (1.0 + torch.exp(-acc))
            ).to(torch.bfloat16)
    flat = conv_out.reshape(batch * rows, channels)
    q_dim = h * k
    v_dim = hv * v
    query = flat[:, :q_dim].reshape(1, batch * rows, h, k).contiguous()
    key = flat[:, q_dim : 2 * q_dim].reshape(
        1, batch * rows, h, k
    ).contiguous()
    value_spec = flat[:, 2 * q_dim :].reshape(
        1, batch * rows, hv, v
    ).contiguous()
    value_tree = flat[:, 2 * q_dim :].reshape(
        batch * rows, hv, v
    ).contiguous()
    g, beta = _gate_incumbent(a, b, A_log, dt_bias)
    zero = torch.zeros(
        (batch, 1, channels), dtype=torch.bfloat16
    )
    source_stage = torch.cat((prior, x, zero), dim=1)
    assert channels == 2 * q_dim + v_dim
    return {
        "query": query,
        "key": key,
        "value_spec": value_spec,
        "value_tree": value_tree,
        "g": g,
        "beta": beta,
        "source_stage": source_stage,
        "conv_tap": conv_out,
    }


def _direct_reference(
    *,
    x: torch.Tensor,
    prior: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    h: int,
    hv: int,
    k: int,
    v: int,
    store_conv_tap: bool,
) -> dict[str, torch.Tensor | None]:
    batch, rows, channels = x.shape
    logical_rows = batch * rows
    q_dim = h * k
    query = torch.empty((1, logical_rows, h, k), dtype=torch.bfloat16)
    key = torch.empty_like(query)
    value_spec = torch.empty(
        (1, logical_rows, hv, v), dtype=torch.bfloat16
    )
    value_tree = torch.empty(
        (logical_rows, hv, v), dtype=torch.bfloat16
    )
    conv_tap = torch.empty_like(x) if store_conv_tap else None
    source_rows = fixed32_descriptorless_sources()
    for request in range(batch):
        source = torch.cat((prior[request], x[request]), dim=0)
        for node, historical in enumerate(source_rows):
            window_rows = (*historical, 3 + node)
            acc = bias.to(torch.float32).clone()
            for tap in range(4):
                value = source[window_rows[tap]].to(torch.bfloat16)
                weight = weights[:, tap].to(torch.bfloat16)
                product = (value * weight).to(torch.bfloat16).to(torch.float32)
                acc = acc + product
            activated_bf16 = (
                acc / (1.0 + torch.exp(0.0 - acc))
            ).to(torch.bfloat16)
            row = request * rows + node
            query[0, row] = activated_bf16[:q_dim].reshape(h, k)
            key[0, row] = activated_bf16[q_dim : 2 * q_dim].reshape(h, k)
            direct_value = activated_bf16[2 * q_dim :].reshape(hv, v)
            value_spec[0, row] = direct_value
            value_tree[row] = direct_value
            if conv_tap is not None:
                conv_tap[request, node] = activated_bf16
    gate_input = a.to(torch.float32) + dt_bias.to(torch.float32)
    positive_softplus = gate_input + torch.log(
        1.0 + torch.exp(-gate_input)
    )
    negative_softplus = torch.log(1.0 + torch.exp(gate_input))
    softplus = torch.where(
        gate_input > 0, positive_softplus, negative_softplus
    )
    softplus = torch.where(gate_input <= 20.0, softplus, gate_input)
    g = -torch.exp(A_log.to(torch.float32)) * softplus
    beta = torch.sigmoid(b.to(torch.float32))
    source_stage = torch.cat(
        (
            prior,
            x,
            torch.zeros((batch, 1, channels), dtype=torch.bfloat16),
        ),
        dim=1,
    )
    return {
        "query": query,
        "key": key,
        "value_spec": value_spec,
        "value_tree": value_tree,
        "g": g,
        "beta": beta,
        "source_stage": source_stage,
        "conv_tap": conv_tap,
    }


def _adversarial_operands() -> dict[str, object]:
    generator = torch.Generator().manual_seed(20260803)
    h, hv, k, v = 1, 2, 4, 4
    channels = 2 * h * k + hv * v
    x = (torch.randn((1, 32, channels), generator=generator) * 2).to(
        torch.bfloat16
    )
    prior = (torch.randn((1, 3, channels), generator=generator) * 2).to(
        torch.bfloat16
    )
    weights = (torch.randn((channels, 4), generator=generator) * 2).to(
        torch.bfloat16
    )
    bias = torch.randn((channels,), generator=generator, dtype=torch.float32)

    # Channel 0 exposes left-to-right FP32 add order. Channel 1 exposes the
    # mandatory BF16 product rounding before conversion to FP32.
    prior[0, :, 0] = torch.tensor([354.0, 784.0, -828.0], dtype=torch.bfloat16)
    x[0, 0, 0] = torch.tensor(-156.0, dtype=torch.bfloat16)
    weights[0] = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
    bias[0] = -940.4055786132812
    prior[0, :, 1] = torch.tensor(
        [-3.515625, -6.96875, 3.015625], dtype=torch.bfloat16
    )
    x[0, 0, 1] = torch.tensor(-8.5625, dtype=torch.bfloat16)
    weights[1] = torch.tensor(
        [0.71875, -2.6875, -8.8125, 0.1484375], dtype=torch.bfloat16
    )
    bias[1] = 0.0

    gate_pattern = torch.tensor(
        [
            float("-inf"),
            -100.0,
            -20.0,
            -0.0,
            0.0,
            20.0,
            100.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    a = gate_pattern.repeat(8)[: 32 * hv].reshape(32, hv).to(torch.bfloat16)
    b = gate_pattern.flip(0).repeat(8)[: 32 * hv].reshape(32, hv).to(
        torch.bfloat16
    )
    A_log = torch.tensor([-2.0, 0.25], dtype=torch.float32)
    below = torch.nextafter(torch.tensor(20.0), torch.tensor(0.0))
    above = torch.nextafter(torch.tensor(20.0), torch.tensor(float("inf")))
    dt_bias = torch.stack((below, above))
    return {
        "x": x,
        "prior": prior,
        "weights": weights,
        "bias": bias,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "h": h,
        "hv": hv,
        "k": k,
        "v": v,
    }


def _valid_layout_operands(*, conv_tap: bool = False) -> dict[str, object]:
    batch = 1
    rows = batch * candidate.ROWS
    x = torch.empty_strided(
        (rows, candidate.CHANNELS),
        (candidate.X_ROW_STRIDE, 1),
        dtype=torch.bfloat16,
    )
    conv_state = torch.empty_strided(
        (2, candidate.CHANNELS, candidate.CONV_STATE_LEN),
        (2097152, 1, candidate.CHANNELS),
        dtype=torch.bfloat16,
    )
    result: dict[str, object] = {
        "batch_size": batch,
        "x": x,
        "conv_state": conv_state,
        "spec_state_indices": torch.zeros(
            (batch, candidate.ROWS), dtype=torch.int32
        ),
        "conv_weights": torch.empty(
            (candidate.CHANNELS, candidate.CONV_WIDTH), dtype=torch.bfloat16
        ),
        "bias": None,
        "a": torch.empty((rows, candidate.NUM_V_HEADS), dtype=torch.bfloat16),
        "b": torch.empty((rows, candidate.NUM_V_HEADS), dtype=torch.bfloat16),
        "A_log": torch.empty((candidate.NUM_V_HEADS,), dtype=torch.float32),
        "dt_bias": torch.empty((candidate.NUM_V_HEADS,), dtype=torch.bfloat16),
        "query": torch.empty(
            (1, rows, candidate.NUM_K_HEADS, candidate.HEAD_K_DIM),
            dtype=torch.bfloat16,
        ),
        "key": torch.empty(
            (1, rows, candidate.NUM_K_HEADS, candidate.HEAD_K_DIM),
            dtype=torch.bfloat16,
        ),
        "value_spec": torch.empty(
            (1, rows, candidate.NUM_V_HEADS, candidate.HEAD_V_DIM),
            dtype=torch.bfloat16,
        ),
        "value_tree": torch.empty(
            (rows, candidate.NUM_V_HEADS, candidate.HEAD_V_DIM),
            dtype=torch.bfloat16,
        ),
        "g": torch.empty((rows, candidate.NUM_V_HEADS), dtype=torch.float32),
        "beta": torch.empty(
            (rows, candidate.NUM_V_HEADS), dtype=torch.float32
        ),
        "source_stage": torch.empty(
            (batch * candidate.SOURCE_ROWS, candidate.CHANNELS),
            dtype=torch.bfloat16,
        ),
        "conv_tap": (
            torch.empty((rows, candidate.CHANNELS), dtype=torch.bfloat16)
            if conv_tap
            else None
        ),
        "expected_device_type": "cpu",
    }
    return result


def test_contract_is_exact_physical32_k64_and_one_launch_per_layer() -> None:
    for batch in (1, 2, 3, 4):
        contract = candidate.fixed32_sfwd_conv_postprep_fusion_contract(
            batch,
            fixed32_mode="hydra27_fixed32",
            tree_parent=list(candidate.FIXED32_PARENT),
            qualification_profile="k64_root",
            draft_vocab_k=65536,
            draft_vocab_root=1,
        )
        assert contract["source_only"] is True
        assert contract["default_off"] is True
        assert contract["production_eligible"] is False
        assert contract["full_graph_qualified"] is True
        assert contract["capture_host_syncs_per_layer"] == 0
        assert contract["physical_rows_per_request"] == 32
        assert contract["launches_per_layer"] == 1
        assert contract["launches_for_all_layers"] == 48
        assert contract["cross_layer_fusion"] is False
        assert contract["conv_product_dtype"] == "bfloat16"
        assert contract["conv_accumulator_dtype"] == "float32"
        assert contract["conv_add_order"] == (
            "bias",
            "tap0",
            "tap1",
            "tap2",
            "tap3",
        )
        assert contract["post_activation_boundary_dtype"] == "bfloat16"
        assert contract["distinct_recurrence_output_storages"] is True
        assert contract["algorithmic_shared_bytes"] == 0
        assert contract["has_reduction"] is False
        assert contract["has_barrier"] is False
        assert contract["candidate_codegen_registers_per_thread"] == 56
        assert contract["source_register_ceiling_per_thread"] == 64
        assert contract["offline_codegen_shared_bytes"] == 0
        assert contract["codegen_registers_verified"] is True
        assert contract["timing_claim"] is False
    with pytest.raises(RuntimeError, match="physical32 K64/root1"):
        candidate.fixed32_sfwd_conv_postprep_fusion_contract(
            1,
            fixed32_mode="hydra27_fixed32",
            tree_parent=candidate.FIXED32_PARENT,
            qualification_profile="full_vocab",
            draft_vocab_k=65536,
            draft_vocab_root=1,
        )
    with pytest.raises(RuntimeError, match="parent vector drifted"):
        candidate.fixed32_sfwd_conv_postprep_fusion_contract(
            1,
            fixed32_mode="hydra27_fixed32",
            tree_parent=(*candidate.FIXED32_PARENT[:-1], 0),
            qualification_profile="k64_root",
            draft_vocab_k=65536,
            draft_vocab_root=1,
        )


def test_exact_direct_algebra_matches_incumbent_bytes_on_adversarial_values() -> None:
    operands = _adversarial_operands()
    incumbent = _incumbent_reference(**operands)
    direct = _direct_reference(**operands, store_conv_tap=True)
    for name in (
        "query",
        "key",
        "value_spec",
        "value_tree",
        "g",
        "beta",
        "source_stage",
        "conv_tap",
    ):
        actual = direct[name]
        assert actual is not None
        assert torch.equal(_bytes(actual), _bytes(incumbent[name])), name
    assert len(
        {
            direct[name].untyped_storage().data_ptr()
            for name in ("query", "key", "value_spec", "value_tree")
        }
    ) == 4


def test_adversaries_detect_product_boundary_and_add_reordering() -> None:
    operands = _adversarial_operands()
    prior = operands["prior"]
    x = operands["x"]
    weights = operands["weights"]
    bias = operands["bias"]
    ordered_values = torch.stack(
        (prior[0, 0, 0], prior[0, 1, 0], prior[0, 2, 0], x[0, 0, 0])
    ).to(torch.float32)
    ordered = bias[0]
    for value in ordered_values:
        ordered = ordered + value
    reversed_acc = bias[0]
    for value in ordered_values.flip(0):
        reversed_acc = reversed_acc + value
    assert ordered.view(torch.int32).item() != reversed_acc.view(torch.int32).item()

    product_values = torch.stack(
        (prior[0, 0, 1], prior[0, 1, 1], prior[0, 2, 1], x[0, 0, 1])
    )
    rounded_acc = torch.tensor(0.0, dtype=torch.float32)
    wide_acc = torch.tensor(0.0, dtype=torch.float32)
    for tap in range(4):
        rounded_product = (
            product_values[tap] * weights[1, tap]
        ).to(torch.bfloat16).to(torch.float32)
        wide_product = product_values[tap].float() * weights[1, tap].float()
        rounded_acc = rounded_acc + rounded_product
        wide_acc = wide_acc + wide_product
    rounded_out = (
        rounded_acc / (1.0 + torch.exp(-rounded_acc))
    ).to(torch.bfloat16)
    wide_out = (wide_acc / (1.0 + torch.exp(-wide_acc))).to(torch.bfloat16)
    assert rounded_out.view(torch.int16).item() != wide_out.view(torch.int16).item()


def test_layout_contract_accepts_exact_surfaces_and_rejects_alias_drift() -> None:
    operands = _valid_layout_operands(conv_tap=True)
    layout = candidate.fixed32_sfwd_conv_postprep_layout_contract(**operands)
    assert layout["conv_state_stride_row"] == 2097152
    assert layout["state_index_bounds"] == (0, 0)
    assert layout["conv_tap"] is True
    assert layout["writable_storages"] == 8
    assert layout["input_aliases_allowed"] is True
    assert layout["input_output_aliases_allowed"] is False

    aliased = dict(operands)
    aliased["key"] = aliased["query"]
    with pytest.raises(ValueError, match="writable_storage_alias"):
        candidate.fixed32_sfwd_conv_postprep_layout_contract(**aliased)

    input_aliased = dict(operands)
    x = input_aliased["x"]
    rows = candidate.ROWS
    input_aliased["query"] = x.as_strided(
        (1, rows, candidate.NUM_K_HEADS, candidate.HEAD_K_DIM),
        (rows * candidate.Q_DIM, candidate.Q_DIM, candidate.HEAD_K_DIM, 1),
    )
    with pytest.raises(ValueError, match="input_output_storage_alias"):
        candidate.fixed32_sfwd_conv_postprep_layout_contract(**input_aliased)

    shared_inputs = dict(operands)
    shared_inputs["b"] = shared_inputs["a"]
    candidate.fixed32_sfwd_conv_postprep_layout_contract(**shared_inputs)


def test_layout_contract_rejects_nonproduction_fp32_dt_bias() -> None:
    operands = _valid_layout_operands()
    operands["dt_bias"] = operands["dt_bias"].to(torch.float32)
    with pytest.raises(ValueError, match="dt_bias_dtype"):
        candidate.fixed32_sfwd_conv_postprep_layout_contract(**operands)


@pytest.mark.parametrize("invalid_index", (-1, 2))
def test_layout_contract_rejects_out_of_range_state_bank_indices(
    invalid_index: int,
) -> None:
    operands = _valid_layout_operands()
    operands["spec_state_indices"][0, -1] = invalid_index
    with pytest.raises(ValueError, match="spec_state_indices_values"):
        candidate.fixed32_sfwd_conv_postprep_layout_contract(**operands)


def test_layout_contract_uses_prevalidated_ssi_without_host_scalar_read() -> None:
    operands = _valid_layout_operands()
    operands["spec_state_indices"][0, -1] = -1
    layout = candidate.fixed32_sfwd_conv_postprep_layout_contract(
        **operands,
        state_indices_prevalidated=True,
    )
    assert layout["state_index_bounds"] is None
    assert layout["state_indices_prevalidated"] is True


@pytest.mark.parametrize(
    ("name", "mutate", "failure"),
    (
        (
            "x",
            lambda tensor: tensor.contiguous(),
            "x_stride",
        ),
        (
            "spec_state_indices",
            lambda tensor: tensor[:, :-1].contiguous(),
            "spec_state_indices_shape",
        ),
        (
            "A_log",
            lambda tensor: tensor.to(torch.bfloat16),
            "A_log_dtype",
        ),
        (
            "value_tree",
            lambda tensor: tensor.transpose(1, 2),
            "value_tree_shape",
        ),
    ),
)
def test_layout_contract_rejects_shape_dtype_and_stride_drift(
    name: str,
    mutate,
    failure: str,
) -> None:
    operands = _valid_layout_operands()
    operands[name] = mutate(operands[name])
    with pytest.raises(ValueError, match=failure):
        candidate.fixed32_sfwd_conv_postprep_layout_contract(**operands)


def test_storage_bound_guard_rejects_truncated_backing_storage() -> None:
    tensor = torch.empty((8,), dtype=torch.float32)
    view = tensor.as_strided((8,), (1,))
    tensor.untyped_storage().resize_(4 * tensor.element_size())
    assert candidate._storage_bound_failure(view) is True


def test_generator_is_deterministic_and_kernel_has_no_conv_intermediate() -> None:
    spec = importlib.util.spec_from_file_location(
        "sfwd_fusion_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    assert generator.generate() == KERNEL_PATH.read_text(encoding="utf-8")

    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel"
    )
    kernel_source = ast.get_source_segment(source, kernel)
    assert kernel_source is not None
    assert "conv_out =" not in kernel_source
    assert "out" not in {argument.arg for argument in kernel.args.args}
    assert "conv_output" not in {argument.arg for argument in kernel.args.args}
    assert "out_batch" not in kernel_source
    assert kernel_source.count("_fr13_store_fixed32_conv_outputs(") == 32
    assert kernel_source.count("tl.load(x_batch") == 32
    assert (
        "bank_row_ok = (bank_row_raw >= 0) & (bank_row_raw < BANK_ROWS)"
        in kernel_source
    )
    assert (
        "tl.maximum(0, tl.minimum(bank_row_raw, BANK_ROWS - 1))"
        in kernel_source
    )
    assert (
        "tl.atomic_xchg(sticky_guard_ok, 0, mask=~bank_row_ok)"
        in kernel_source
    )
    assert "prior_base = conv_state + bank_row_raw" not in kernel_source
    assert kernel_source.count(".to(tl.bfloat16).to(tl.float32)") == 128
    assert "tl.sum" not in kernel_source
    assert "tl.dot" not in kernel_source
    assert "barrier" not in kernel_source
    assert "pid_task < channel_tasks" in kernel_source
    assert "pid_n = pid_task - channel_tasks" in kernel_source
    assert "gate_input <= SOFTPLUS_THRESHOLD" in kernel_source
    store_helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_store_fixed32_conv_outputs"
    )
    store_source = ast.get_source_segment(source, store_helper)
    assert store_source is not None
    assert "activated_bf16 = activated.to(tl.bfloat16)" in store_source
    assert "tl.store(value_spec + value_offset" in store_source
    assert "tl.store(value_tree + value_offset" in store_source


def test_launcher_requires_opaque_capture_binding_and_has_one_launch() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    launcher = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "launch_fixed32_sfwd_conv_postprep_fusion"
    )
    launcher_source = ast.get_source_segment(source, launcher)
    assert launcher_source is not None
    assert "source_only_qualification is not True" in launcher_source
    assert "physical32_guarded" not in launcher_source
    assert "capture_binding is None and capturing" in launcher_source
    assert "_validate_capture_binding(" in launcher_source
    assert "state_indices_prevalidated=capture_binding is not None" in launcher_source
    assert 'capture_binding.committer_state["sticky_guard_ok"]' in launcher_source
    assert "BANK_ROWS=int(conv_state.size(0))" in launcher_source
    assert "CAPTURE_GUARD=capture_binding is not None" in launcher_source
    assert "grid = (int(batch_size), channel_tasks + ROWS)" in launcher_source
    assert launcher_source.count(
        "_fr13_fixed32_sfwd_conv_postprep_fusion_kernel[grid]("
    ) == 1
    assert "num_warps=num_warps" in launcher_source
    patcher_source = PATCHER_PATH.read_text(encoding="utf-8")
    assert "preseed_fixed32_sfwd_conv_postprep_capture_bindings" in patcher_source
    assert "capture_binding=_fr13_conv_postprep_binding" in patcher_source
    assert "if any(existing_graph_caches):" in source
    assert "partially preseeded" in source


def test_static_ledger_counts_exact_bytes_and_launches_without_timing() -> None:
    b1 = candidate.fixed32_sfwd_conv_postprep_static_ledger(1)
    assert b1["per_layer_bytes"] == {
        "incumbent_conv_intermediate_write": 655360,
        "incumbent_rearrange_read": 655360,
        "incumbent_fused_postprep_read": 655360,
        "incumbent_dead_normalized_qk_write": 262144,
        "direct_recurrence_qkv_write": 655360,
        "direct_value_tree_write": 393216,
        "direct_g_beta_write": 12288,
        "unchanged_commit_source_stage_write": 737280,
        "optional_conv_tap_write": 0,
        "mandatory_direct_writes": 1798144,
        "logical_traffic_removed": 2228224,
    }
    assert b1["all_layer_bytes"] == {
        "conv_intermediate_write_removed": 31457280,
        "conv_intermediate_reads_removed": 62914560,
        "dead_normalized_qk_write_removed": 12582912,
        "logical_traffic_removed": 106954752,
    }
    assert b1["launches"] == {
        "incumbent_per_layer": 5,
        "candidate_per_layer": 1,
        "incumbent_all_layers": 240,
        "candidate_all_layers": 48,
        "removed_all_layers": 192,
    }
    assert "timing" in b1["excludes"]
    b4 = candidate.fixed32_sfwd_conv_postprep_static_ledger(4)
    for name, value in b1["per_layer_bytes"].items():
        assert b4["per_layer_bytes"][name] == 4 * value
    tap = candidate.fixed32_sfwd_conv_postprep_static_ledger(
        1, store_conv_tap=True
    )
    assert tap["per_layer_bytes"]["optional_conv_tap_write"] == 655360
    assert tap["all_layer_bytes"]["conv_intermediate_write_removed"] == 0


def test_byte_gate_requires_exact_arm_and_authenticated_task(
    tmp_path: Path,
) -> None:
    enabled = tmp_path / "enabled"
    event = tmp_path / "event"
    environ = {"FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB": "1"}

    with pytest.raises(RuntimeError, match="byte-gate arm is missing"):
        candidate.fixed32_sfwd_conv_postprep_gate_control(
            fixed32_mode="hydra27_fixed32",
            environ=environ,
            enabled_path=str(enabled),
            event_path=str(event),
        )

    enabled.write_bytes(b"1\n")
    enabled.chmod(0o400)
    assert candidate.fixed32_sfwd_conv_postprep_gate_control(
        fixed32_mode="hydra27_fixed32",
        environ=environ,
        enabled_path=str(enabled),
        event_path=str(event),
    ) == (True, None)

    event.write_text(candidate.TASK_MARKER + "\n", encoding="ascii")
    event.chmod(0o444)
    assert candidate.fixed32_sfwd_conv_postprep_gate_control(
        fixed32_mode="hydra27_fixed32",
        environ=environ,
        enabled_path=str(enabled),
        event_path=str(event),
    ) == (True, candidate.TASK_MARKER)

    event.chmod(0o644)
    with pytest.raises(RuntimeError, match="marker mode must be 0444"):
        candidate.fixed32_sfwd_conv_postprep_gate_control(
            fixed32_mode="hydra27_fixed32",
            environ=environ,
            enabled_path=str(enabled),
            event_path=str(event),
        )


def test_byte_gate_emits_48_layer_pass_and_mismatch_is_sticky(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    commit = "1" * 40
    manifest, manifest_sha256 = _byte_gate_source_manifest(tmp_path, commit)
    records = tmp_path / "records.jsonl"
    live_pass = tmp_path / "live_pass.json"
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_BYTE_AB_PATH", str(records)
    )
    monkeypatch.setenv(
        "FR13_FIXED32_SFWD_CONV_POSTPREP_PASS_PATH", str(live_pass)
    )
    monkeypatch.setenv(
        "FR13_DRAFT_VOCAB_BLOCKS",
        str(ROOT / "scripts" / "fr13_dvk_subset_blocks.json"),
    )
    monkeypatch.setenv("FR13_DRAFT_VOCAB_ROOT", "1")
    monkeypatch.setenv("FR13_DRAFT_VOCAB_K", "65536")
    monkeypatch.setenv("FR13_FA2_QROW16_PRODUCTION", "1")
    monkeypatch.setenv("FR13_FA2_QROW16_SO_SHA256", candidate.QROW16_FA2_SHA256)
    monkeypatch.setenv(
        "FR13_FA2_QROW16_LIVE_PASS_SHA256",
        candidate.QROW16_LIVE_PASS_SHA256,
    )
    reference = torch.tensor([1.0, -2.0], dtype=torch.bfloat16)

    def compare(layer_key: int, candidate_query: torch.Tensor) -> dict[str, object]:
        return candidate.fixed32_sfwd_conv_postprep_byte_gate(
            fixed32_mode="hydra27_fixed32",
            task_marker=candidate.TASK_MARKER,
            layer_prefix=f"model.layers.{layer_key}",
            layer_key=layer_key,
            batch_size=1,
            reference_query=reference,
            candidate_query=candidate_query,
            reference_key=reference,
            candidate_key=reference.clone(),
            reference_value_spec=reference,
            candidate_value_spec=reference.clone(),
            reference_value_tree=reference,
            candidate_value_tree=reference.clone(),
            reference_g=reference,
            candidate_g=reference.clone(),
            reference_beta=reference,
            candidate_beta=reference.clone(),
            reference_source_stage=reference,
            candidate_source_stage=reference.clone(),
            source_manifest_path=str(manifest),
            expected_source_manifest_sha256=manifest_sha256,
            expected_source_commit=commit,
        )

    candidate._BYTE_GATE_STATE.update(
        task_marker=None,
        source_binding=None,
        passed={},
        attempts={},
        failed=False,
    )
    for layer_key in range(1, 49):
        record = compare(layer_key, reference.clone())
        assert record["zero_diff"] is True

    payload = json.loads(live_pass.read_text(encoding="ascii"))
    assert payload["layer_count"] == 48
    assert payload["comparisons"] == 48 * len(candidate.BYTE_SURFACES)
    assert payload["compared_byte_surfaces"] == list(candidate.BYTE_SURFACES)
    assert payload["reference_decision"] == "serve_incumbent"
    assert payload["candidate_decision"] == "shadow_only"
    assert len(records.read_text(encoding="ascii").splitlines()) == 48

    candidate._BYTE_GATE_STATE.update(
        task_marker=None,
        source_binding=None,
        passed={},
        attempts={},
        failed=False,
    )
    live_pass.unlink()
    mismatch = reference.clone()
    mismatch[0] = 3.0
    assert compare(1, mismatch)["zero_diff"] is False
    for layer_key in range(1, 49):
        assert compare(layer_key, reference.clone())["zero_diff"] is True
    assert not live_pass.exists()


def test_sanitized_artifact_binds_sources_ledgers_and_resource_gate() -> None:
    manifest = json.loads((ARTIFACT / "source_manifest.json").read_text())
    assert manifest["base_commit"] == "c49c8eb5370e4d4035aceffaa8476aea31f921f5"
    assert manifest["source_commit"] == "7f46f69a76ac1b0a7429ec0928b641bbeee2efb2"
    for relative, expected in manifest["files"].items():
        # The artifact's self-test entry was captured by its follow-up commit.
        snapshot_commit = (
            "711c08551de2e5eabf9788ad44002e5b0a2564db"
            if relative == "tests/test_fr13_fixed32_sfwd_conv_postprep_fusion.py"
            else manifest["source_commit"]
        )
        raw = subprocess.run(
            [
                "git",
                "-C",
                str(ROOT),
                "show",
                f"{snapshot_commit}:{relative}",
            ],
            check=True,
            capture_output=True,
        ).stdout
        assert len(raw) == expected["bytes"]
        assert hashlib.sha256(raw).hexdigest() == expected["sha256"]

    ledgers = json.loads((ARTIFACT / "static_ledger.json").read_text())
    assert ledgers["b1"] == json.loads(
        json.dumps(candidate.fixed32_sfwd_conv_postprep_static_ledger(1))
    )
    assert ledgers["b4"] == json.loads(
        json.dumps(candidate.fixed32_sfwd_conv_postprep_static_ledger(4))
    )
    codegen = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert codegen["offline_only"] is True
    assert codegen["timing_claim"] is False
    assert codegen["compile_contract"]["capture_guard"] is True
    assert codegen["compile_contract"]["bank_rows_fixture"] == 257
    assert codegen["resource_gate"]["max_registers_per_thread"] == 64
    assert set(codegen["profiles"]) == {"b1", "b4", "b1_tap", "b4_tap"}
    assert codegen["profiles"]["b1"]["registers_per_thread"] == 64
    assert codegen["profiles"]["b4"]["registers_per_thread"] == 64
    assert codegen["profiles"]["b1_tap"]["registers_per_thread"] == 56
    assert codegen["profiles"]["b4_tap"]["registers_per_thread"] == 56
    for profile in codegen["profiles"].values():
        assert profile["registers_per_thread"] <= 64
        assert profile["stack_bytes"] == 0
        assert profile["local_bytes"] == 0
        assert profile["shared_bytes"] == 0
    readme = (ARTIFACT / "README.md").read_text(encoding="utf-8")
    assert "default-off eager and FULL-capture wiring complete" in readme
    assert "not GPU measured" in readme
    assert "No device API" in (ARTIFACT / "verification.md").read_text()
