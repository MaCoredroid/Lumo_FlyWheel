from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumo_flywheel_serving.fr10_tree_conv import (  # noqa: E402
    flat_causal_conv1d_reference,
    tree_causal_conv1d_reference,
)


TREE_PARENT = [-1, 0, 1, 1, 2, 2, 4, 4, 6, 6]
PATH0_NODES = [0, 1, 2, 4, 6, 8]
LEAF_NODES = [3, 5, 7, 9]


def _path_to(parent: list[int], node: int) -> list[int]:
    path = []
    cur = node
    while cur >= 0:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _tree_scan_reference(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    parent: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    states = torch.empty_like(x)
    for node, par in enumerate(parent):
        prev = initial_state if par < 0 else states[par]
        state = prev + x[node]
        states[node] = state
        out[node] = state
    return out, states


def _flat_scan_reference(
    x: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    parent = [-1] + [idx - 1 for idx in range(1, int(x.shape[0]))]
    return _tree_scan_reference(x, initial_state, parent)


def _native_linear_scan_state(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    path: list[int],
) -> torch.Tensor:
    state = initial_state.clone()
    for node in path:
        state = state + x[node]
    return state


def _native_linear_conv_state(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    path: list[int],
) -> torch.Tensor:
    state = initial_state.clone()
    width = int(state.shape[1]) + 1
    for node in path:
        if width > 2:
            state[:, : width - 2] = state[:, 1 : width - 1].clone()
        state[:, width - 2] = x[node]
    return state


def _materialize_temporal_conv_rows(
    x: torch.Tensor,
    initial_state_row: torch.Tensor,
    parent: list[int],
) -> torch.Tensor:
    rows = []
    state_len = int(initial_state_row.shape[1])
    for node in range(len(parent)):
        path = _path_to(parent, node)
        source = torch.cat((initial_state_row.transpose(0, 1), x[path]), dim=0)
        idx = len(path) + torch.arange(state_len)
        rows.append(source.index_select(0, idx).transpose(0, 1))
    return torch.stack(rows)


def _bad_global_accept_temporal_conv_rows(
    x: torch.Tensor,
    initial_state_row: torch.Tensor,
    parent: list[int],
    *,
    accepted_offset: int,
) -> torch.Tensor:
    rows = []
    state_len = int(initial_state_row.shape[1])
    bad_idx = accepted_offset + 1 + torch.arange(state_len)
    for node in range(len(parent)):
        path = _path_to(parent, node)
        source = torch.cat((initial_state_row.transpose(0, 1), x[path]), dim=0)
        rows.append(source.index_select(0, bad_idx).transpose(0, 1))
    return torch.stack(rows)


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    node_count, dim, width = len(TREE_PARENT), 5, 4
    x = torch.arange(node_count * dim, dtype=torch.float32).reshape(node_count, dim) / 17.0
    scan0 = torch.arange(dim, dtype=torch.float32) / 11.0
    conv0 = torch.arange(dim * (width - 1), dtype=torch.float32).reshape(dim, width - 1) / 13.0
    weight = torch.arange(dim * width, dtype=torch.float32).reshape(dim, width) / 19.0
    bias = torch.arange(dim, dtype=torch.float32) / 23.0
    return x, scan0, conv0, weight, bias


def test_sibling_leaf_must_not_mutate_trunk_conv_scan_or_commit() -> None:
    x, scan0, conv0, weight, bias = _make_inputs()
    base_conv_out, base_conv_states = tree_causal_conv1d_reference(
        x,
        conv0,
        weight,
        bias,
        TREE_PARENT,
        activation=None,
    )
    base_scan_out, base_scan_states = _tree_scan_reference(x, scan0, TREE_PARENT)
    accepted_node = PATH0_NODES[-1]
    base_committed_conv = base_conv_states[accepted_node].clone()
    base_committed_scan = base_scan_states[accepted_node].clone()

    for leaf in LEAF_NODES:
        perturbed = x.clone()
        perturbed[leaf] = perturbed[leaf] + 10_000.0
        conv_out, conv_states = tree_causal_conv1d_reference(
            perturbed,
            conv0,
            weight,
            bias,
            TREE_PARENT,
            activation=None,
        )
        scan_out, scan_states = _tree_scan_reference(perturbed, scan0, TREE_PARENT)

        assert torch.equal(conv_out[PATH0_NODES], base_conv_out[PATH0_NODES])
        assert torch.equal(conv_states[accepted_node], base_committed_conv)
        assert torch.equal(scan_out[PATH0_NODES], base_scan_out[PATH0_NODES])
        assert torch.equal(scan_states[accepted_node], base_committed_scan)


def test_sibling_leaf_gate_powered_stock_flat_controls_fail() -> None:
    x, scan0, conv0, weight, bias = _make_inputs()
    base_flat_conv, _ = flat_causal_conv1d_reference(
        x,
        conv0,
        weight,
        bias,
        activation=None,
    )
    base_flat_scan, base_flat_states = _flat_scan_reference(x, scan0)

    perturbed = x.clone()
    perturbed[3] = perturbed[3] + 10_000.0
    flat_conv, _ = flat_causal_conv1d_reference(
        perturbed,
        conv0,
        weight,
        bias,
        activation=None,
    )
    flat_scan, flat_states = _flat_scan_reference(perturbed, scan0)

    assert not torch.equal(flat_conv[4], base_flat_conv[4])
    assert not torch.equal(flat_scan[4], base_flat_scan[4])
    assert not torch.equal(flat_states[PATH0_NODES[-1]], base_flat_states[PATH0_NODES[-1]])


def test_cross_step_commit_parity_equals_native_over_accepted_linear_tokens() -> None:
    x, scan0, conv0, weight, bias = _make_inputs()
    _, tree_conv_states = tree_causal_conv1d_reference(
        x,
        conv0,
        weight,
        bias,
        TREE_PARENT,
        activation=None,
    )
    _, tree_scan_states = _tree_scan_reference(x, scan0, TREE_PARENT)

    for accepted_node in [*PATH0_NODES, *LEAF_NODES]:
        accepted_path = _path_to(TREE_PARENT, accepted_node)
        native_conv = _native_linear_conv_state(x, conv0, accepted_path)
        native_scan = _native_linear_scan_state(x, scan0, accepted_path)

        assert torch.equal(tree_conv_states[accepted_node], native_conv)
        assert torch.equal(tree_scan_states[accepted_node], native_scan)


def test_cross_step_commit_parity_powered_flat_commit_fails_on_branch_leaf() -> None:
    x, scan0, conv0, weight, bias = _make_inputs()
    _, flat_conv_states = flat_causal_conv1d_reference(
        x,
        conv0,
        weight,
        bias,
        activation=None,
    )
    _, flat_scan_states = _flat_scan_reference(x, scan0)
    accepted_node = 5
    accepted_path = _path_to(TREE_PARENT, accepted_node)

    assert not torch.equal(
        flat_conv_states[accepted_node],
        _native_linear_conv_state(x, conv0, accepted_path),
    )
    assert not torch.equal(
        flat_scan_states[accepted_node],
        _native_linear_scan_state(x, scan0, accepted_path),
    )


def test_temporal_conv_rows_use_each_node_path_length_not_global_accept_offset() -> None:
    x, _, _, _, _ = _make_inputs()
    state_len = 12
    initial_state_row = (
        torch.arange(5 * state_len, dtype=torch.float32).reshape(5, state_len) / 7.0
    )
    rows = _materialize_temporal_conv_rows(x, initial_state_row, TREE_PARENT)

    for node in range(len(TREE_PARENT)):
        path = _path_to(TREE_PARENT, node)
        source = torch.cat((initial_state_row.transpose(0, 1), x[path]), dim=0)
        expected = source[len(path) : len(path) + state_len].transpose(0, 1)
        assert torch.equal(rows[node], expected)

    # This is the d0ac0862 bug: a single accepted-path offset was reused for
    # every node row. Shallow sibling rows then index past their own source.
    try:
        _bad_global_accept_temporal_conv_rows(
            x,
            initial_state_row,
            TREE_PARENT,
            accepted_offset=5,
        )
    except IndexError:
        pass
    else:
        raise AssertionError("global accepted offset should not materialize all rows")
