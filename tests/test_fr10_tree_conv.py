from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_tree_conv import (
    flat_causal_conv1d_reference,
    tree_causal_conv1d_reference,
)


def _serial_path_conv(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    path: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    outputs = []
    for node in path:
        out, states = tree_causal_conv1d_reference(
            x[node : node + 1],
            state,
            weight,
            bias,
            [-1],
            activation="silu",
        )
        outputs.append(out[0])
        state = states[0]
    return torch.stack(outputs), state


def test_tree_causal_conv_matches_serial_per_path() -> None:
    torch.manual_seed(10)
    # Runtime-sorted caterpillar with root plus draft nodes:
    # 0=root, 1=spine depth1, 2=branch depth1, 3=spine depth2,
    # 4=branch depth2, 5=spine depth3, 6=branch depth3.
    parent = [-1, 0, 0, 1, 1, 3, 3]
    x = torch.randn(len(parent), 8, dtype=torch.float32)
    initial_state = torch.randn(8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.float32)

    tree_out, tree_states = tree_causal_conv1d_reference(
        x,
        initial_state,
        weight,
        bias,
        parent,
        activation="silu",
    )

    paths = {
        5: [0, 1, 3, 5],
        6: [0, 1, 3, 6],
        2: [0, 2],
        4: [0, 1, 4],
    }
    for leaf, path in paths.items():
        serial_out, serial_state = _serial_path_conv(
            x,
            initial_state,
            weight,
            bias,
            path,
        )
        assert torch.allclose(tree_out[path], serial_out, atol=1e-6, rtol=0)
        assert torch.allclose(tree_states[leaf], serial_state, atol=1e-6, rtol=0)


def test_flattened_causal_conv_negative_control_leaks_sibling() -> None:
    torch.manual_seed(11)
    parent = [-1, 0, 0, 1, 1, 3, 3]
    x = torch.randn(len(parent), 8, dtype=torch.float32)
    # Make branch node 2 very large so flattened order visibly contaminates
    # spine node 3, while ancestry-aware conv excludes it.
    x[2] = 100.0
    initial_state = torch.randn(8, 3, dtype=torch.float32)
    weight = torch.randn(8, 4, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.float32)

    tree_out, _ = tree_causal_conv1d_reference(
        x,
        initial_state,
        weight,
        bias,
        parent,
        activation="silu",
    )
    flat_out, _ = flat_causal_conv1d_reference(
        x,
        initial_state,
        weight,
        bias,
        activation="silu",
    )

    assert (flat_out[3] - tree_out[3]).abs().max().item() > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_linear_tree_causal_conv_matches_stock_vllm_conv() -> None:
    """Linear tree path must reduce to vLLM's stock flat causal conv.

    This is the gate that catches state/weight orientation regressions on the
    public path. It runs only in the cu130/vLLM environment where the production
    Triton causal-conv op is installed.
    """
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_fn,
    )

    torch.manual_seed(12)
    n, dim, width = 5, 16, 4
    device = torch.device("cuda")
    x_nodes = torch.randn(n, dim, device=device, dtype=torch.float32).contiguous()
    # vLLM causal_conv1d_fn expects channel-last storage for a [dim, tokens]
    # logical tensor, so transpose without making it contiguous.
    x_stock = x_nodes.T
    weight = torch.randn(dim, width, device=device, dtype=torch.float32)
    bias = torch.randn(dim, device=device, dtype=torch.float32)
    conv_state = torch.zeros(2, dim, width - 1, device=device, dtype=torch.float32)
    conv_state[1] = torch.randn(dim, width - 1, device=device, dtype=torch.float32)
    conv_state0 = conv_state.clone()

    stock_out = causal_conv1d_fn(
        x_stock,
        weight,
        bias,
        conv_state,
        torch.tensor([0, n], device=device, dtype=torch.int32),
        cache_indices=torch.tensor([1], device=device, dtype=torch.int32),
        has_initial_state=torch.tensor([True], device=device),
        activation="silu",
        null_block_id=0,
        validate_data=True,
    ).T.contiguous()
    tree_out, tree_states = tree_causal_conv1d_reference(
        x_nodes,
        conv_state0[1],
        weight,
        bias,
        [-1, 0, 1, 2, 3],
        activation="silu",
    )
    torch.cuda.synchronize()

    assert torch.allclose(stock_out, tree_out, atol=1e-6, rtol=0)
    assert torch.equal(conv_state[1], tree_states[-1])
