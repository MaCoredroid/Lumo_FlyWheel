"""FR13 replay-route pure-torch fp32 reference (CPU bitwise-identity proofs).

This module mirrors, in pure torch fp32 with the EXACT op order, the shared
@triton.jit per-node body ``_gdn_node_step`` and the two state lineages built
on it in ``fr10_gdn_tree_kernel.py``:

- the tree SCAN chain (``_tree_gdn_kernel``): h_cache register checkpoints,
  parent handoff through ``tl.sum(tl.where(offs_n == j, h_cache, 0.0), 0)``
  (which flips ``-0.0`` to ``+0.0`` exactly once per edge), root consumes h0
  with NO normalization;
- the accepted-path REPLAY chain (``_tree_gdn_replay_kernel``): sequential
  re-execution of root + accepted path from stored activations, with
  ``state = state + 0.0`` per post-root parent-handoff edge, publishing to
  linear bank columns (root always refreshes column 0 for the zero-accept
  path).

WHAT THE CPU TESTS ON THIS MODULE PROVE: the replay restructuring (store
small activations, re-execute the identical instruction sequence, emulate the
masked-sum handoff with ``+ 0.0``) is BITWISE identical to the scan chain on
IEEE-754 fp32, including -0.0 handoffs, empty accepted paths, and
bf16-roundtripped activations. Replay does NOT reconstruct/reassociate (the
delta-only failure class); it re-executes.

WHAT THEY DO NOT PROVE (GPU-gated): that the two Triton compilations of the
shared body produce identical machine code (FMA contraction/scheduling per
unrolled instance), and that the live serving wiring is correct. Those are
decided by the one-time byte A/B on captured payloads and the live gates
listed in FR13_REPLAY_ROUTE_BUILD.md.

All inputs are fp32 tensors; per-head layout: state ``(DIM_V, DIM_K)``,
k ``(DIM_K,)`` PRE-l2norm, v ``(DIM_V,)``, raw_a/raw_b/a_log/dt_bias scalars
(0-dim fp32 tensors).
"""

from __future__ import annotations

import torch


def _require_fp32(name: str, t: torch.Tensor) -> None:
    if t.dtype != torch.float32:
        raise ValueError(f"{name} must be fp32, got {t.dtype}")


def gdn_node_step_ref(
    state: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    q: torch.Tensor | None = None,
    output_scale: float = 1.0,
    use_qk_l2norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One GDN rank-1 node update, exact `_gdn_node_step` op order.

    RAW_GATING=True basis (the serving configuration): the native
    gate-folding recompute (softplus with the x<=20 branch, sigmoid,
    -exp(A_log)*softplus), k l2norm via rsqrt(sum + 1e-6), then the
    multiply-then-rank-1 update. q-side ops never touch state; q=None skips
    them (the replay kernel's discarded-out case).
    """
    for name, t in (
        ("state", state),
        ("k", k),
        ("v", v),
        ("raw_a", raw_a),
        ("raw_b", raw_b),
        ("a_log", a_log),
        ("dt_bias", dt_bias),
    ):
        _require_fp32(name, t)
    # RAW_GATING recompute (kernel: x = raw_a + dt_bias; softplus x<=20
    # branch; b_g = -exp(A_log) * softplus; b_beta = sigmoid(raw_b)).
    x = raw_a + dt_bias
    softplus_x = torch.where(
        x <= 20.0,
        torch.log(1.0 + torch.exp(x)),
        x,
    )
    b_g = -torch.exp(a_log) * softplus_x
    b_beta = torch.sigmoid(raw_b)
    # USE_QK_L2NORM_IN_KERNEL
    if use_qk_l2norm:
        if q is not None:
            q = q * torch.rsqrt((q * q).sum() + 1e-6)
        k = k * torch.rsqrt((k * k).sum() + 1e-6)
    if q is not None:
        q = q * output_scale
    # The rank-1 update (kernel :"state_i *= tl.exp(b_g)" onward).
    state = state * torch.exp(b_g)
    v = v - (state * k.unsqueeze(0)).sum(dim=1)
    v = v * b_beta
    state = state + v.unsqueeze(1) * k.unsqueeze(0)
    out = None
    if q is not None:
        out = (state * q.unsqueeze(0)).sum(dim=1)
    return state, out


def handoff_ref(h_cache: torch.Tensor, j: int) -> torch.Tensor:
    """Parent handoff: tl.sum(tl.where(offs_n == j, h_cache, 0.0), axis=0).

    Emulated literally: a zeros stack with only row j populated, then summed
    over the node axis. The +0.0 partial sums flip -0.0 to +0.0 exactly as
    the kernel reduction does.
    """
    sel = torch.zeros_like(h_cache)
    sel[j] = h_cache[j]
    return sel.sum(dim=0)


def strict_ancestor_mask(parents: list[int], n_pad: int) -> list[list[bool]]:
    """Strict ancestry mask, mirroring Tree.masks / the patcher's builder."""
    n = len(parents)
    strict = [[False] * n_pad for _ in range(n_pad)]
    for i in range(n):
        cur = parents[i]
        while cur >= 0:
            strict[i][cur] = True
            cur = parents[cur]
    return strict


def tree_scan_chain_ref(
    h0: torch.Tensor,
    parents: list[int],
    k: torch.Tensor,
    v: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    output_scale: float = 1.0,
) -> list[torch.Tensor]:
    """Per-head mirror of the `_tree_gdn_kernel` scan chain.

    Returns the per-node post-update states (the h_cache checkpoints, equal
    to what STORE_NODE_STATES=True would export). Node i starts from h0,
    then every strict ancestor j < i overwrites the running state with the
    handoff of h_cache[j]; the LAST surviving ancestor is the immediate
    parent (parents[i] < i by construction), exactly like the kernel's
    tl.where chain. The root (i with no ancestors) consumes h0 unnormalized.
    """
    n = len(parents)
    n_pad = 1 << max(0, (n - 1).bit_length())
    strict = strict_ancestor_mask(parents, n_pad)
    h_cache: list[torch.Tensor | None] = [None] * n
    states: list[torch.Tensor] = []
    for i in range(n):
        state_i = h0
        for j in range(i):
            if strict[i][j]:
                state_i = handoff_ref_from_cache(h_cache, j)
        state_i, _ = gdn_node_step_ref(
            state_i,
            k[i],
            v[i],
            raw_a[i],
            raw_b[i],
            a_log,
            dt_bias,
            output_scale=output_scale,
        )
        h_cache[i] = state_i
        states.append(state_i)
    return states


def handoff_ref_from_cache(h_cache: list[torch.Tensor | None], j: int) -> torch.Tensor:
    """handoff_ref over a sparse python h_cache (only computed rows kept)."""
    h_j = h_cache[j]
    if h_j is None:
        raise ValueError(f"handoff from uncomputed node {j}")
    stack = torch.zeros((2, *h_j.shape), dtype=h_j.dtype)
    stack[0] = h_j
    return stack.sum(dim=0)


def accepted_path_to_root_chain(parents: list[int], node: int) -> list[int]:
    """Root-to-node gdn path EXCLUDING the root (committer convention)."""
    path: list[int] = []
    cur = node
    while cur > 0:
        path.append(cur)
        cur = parents[cur]
    if cur != 0:
        raise ValueError(f"node {node} does not chain to root")
    return list(reversed(path))


def tree_replay_chain_ref(
    h0: torch.Tensor,
    path: list[int],
    k: torch.Tensor,
    v: torch.Tensor,
    raw_a: torch.Tensor,
    raw_b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    output_scale: float = 1.0,
) -> tuple[list[torch.Tensor], dict[int, torch.Tensor]]:
    """Per-head mirror of `_tree_gdn_replay_kernel` for one request.

    ``path`` is the committed accepted path EXCLUDING the root (gdn node ids
    >= 1, each node's parent the previous entry, path[0]'s parent the root).
    Replays [root] + path; applies ``state = state + 0.0`` per post-root
    edge (the scan's masked-sum -0.0 flip); root gets NO normalization of
    h0. Returns (per-step states, final linear bank column writes). The root
    state goes to column 0 unconditionally (the zero-accept row-0 publish);
    step t >= 1 stores to column t-1 (overwriting the root's column-0 write
    when the path is non-empty, matching the legacy publish+remap ordering).
    """
    state = h0
    states: list[torch.Tensor] = []
    bank_writes: dict[int, torch.Tensor] = {}
    for t in range(len(path) + 1):
        node = 0 if t == 0 else path[t - 1]
        if t >= 1:
            state = state + 0.0
        state, _ = gdn_node_step_ref(
            state,
            k[node],
            v[node],
            raw_a[node],
            raw_b[node],
            a_log,
            dt_bias,
            output_scale=output_scale,
        )
        states.append(state)
        bank_writes[0 if t == 0 else t - 1] = state
    return states, bank_writes


def legacy_consumed_columns(
    scan_states: list[torch.Tensor], path: list[int]
) -> dict[int, torch.Tensor]:
    """Final linear-column contents the LEGACY route exposes to consumers.

    Legacy publishes node i's state to column i, then the next-step ssm
    remap copies column t <- column path[t] for t < len(path). Zero-accept
    leaves column 0 = node-0 (root) state. Only columns 0..max(len-1, 0) are
    consumed (the next h0 read clamps accepted_len-1 to 0).
    """
    if not path:
        return {0: scan_states[0]}
    return {t: scan_states[path[t]] for t in range(len(path))}
