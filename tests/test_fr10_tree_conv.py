from __future__ import annotations

import sys
from pathlib import Path

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

