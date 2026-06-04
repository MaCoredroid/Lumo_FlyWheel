"""Tree-aware causal-conv reference for FR10 GDN.

This is the convolution analogue of the FR10 tree delta-rule proof: each node's
causal convolution may see only its root-path ancestors, never flattened
siblings. The production Triton path can be optimized separately; this module is
kept small and explicit so tests can catch sibling leakage.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _activate(x: torch.Tensor, activation: str | bool | None) -> torch.Tensor:
    if activation in (True, "silu", "swish"):
        return torch.nn.functional.silu(x)
    return x


def tree_causal_conv1d_reference(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    parent: Sequence[int],
    *,
    activation: str | bool | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run causal conv over a tree using only ancestry windows.

    Args:
        x: ``[N, D]`` node inputs in runtime topological order.
        initial_state: ``[D, W-1]`` prior conv state for the committed prefix.
        weight: ``[D, W]`` causal-conv weights, matching vLLM's layout.
        bias: optional ``[D]`` bias.
        parent: length ``N`` parent ids; roots use ``-1``.

    Returns:
        ``(out, states)`` where ``out`` is ``[N, D]`` and ``states`` is
        ``[N, D, W-1]`` containing the post-node conv state for each node.
    """
    if x.ndim != 2:
        raise ValueError("x must have shape [N, D]")
    if initial_state.ndim != 2:
        raise ValueError("initial_state must have shape [D, W-1]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [D, W]")
    n, dim = x.shape
    if len(parent) != n:
        raise ValueError("parent length must match N")
    if weight.shape[0] != dim or initial_state.shape[0] != dim:
        raise ValueError("dimension mismatch")
    width = int(weight.shape[1])
    if initial_state.shape[1] < width - 1:
        raise ValueError("initial_state is shorter than W-1")

    state0 = initial_state[:, : width - 1].to(torch.float32)
    x_f = x.to(torch.float32)
    w_f = weight.to(torch.float32)
    bias_f = None if bias is None else bias.to(torch.float32)

    out = torch.empty((n, dim), device=x.device, dtype=torch.float32)
    states = torch.empty((n, dim, width - 1), device=x.device, dtype=torch.float32)
    for node, par in enumerate(parent):
        history = state0 if int(par) < 0 else states[int(par)]
        acc = x_f[node] * w_f[:, width - 1]
        if bias_f is not None:
            acc = acc + bias_f
        for col in range(width - 1):
            acc = acc + history[:, col] * w_f[:, col]
        out[node] = _activate(acc, activation)
        if width > 1:
            if width > 2:
                states[node, :, : width - 2] = history[:, 1 : width - 1]
            states[node, :, width - 2] = x_f[node]
    return out.to(x.dtype), states


def flat_causal_conv1d_reference(
    x: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    activation: str | bool | None = "silu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Negative-control linearized causal conv over flattened node order."""
    parent = [-1] + [idx - 1 for idx in range(1, int(x.shape[0]))]
    return tree_causal_conv1d_reference(
        x,
        initial_state,
        weight,
        bias,
        parent,
        activation=activation,
    )
