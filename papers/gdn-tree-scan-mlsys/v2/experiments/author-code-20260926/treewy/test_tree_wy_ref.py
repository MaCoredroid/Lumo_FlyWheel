# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for the tree-WY gated-delta-rule reference and its parity with
vLLM's chain kernel ``chunk_gated_delta_rule`` .

  * CPU tests (no GPU): tree-WY single-solve form == naive per-node recurrence,
    for outputs AND every reconstructed (committed) state, over random trees.
  * GPU test (B200 etc.): on a CHAIN, the reference == chunk_gated_delta_rule
    outputs and final state.  Skipped without CUDA.

Run CPU only:    pytest tests/kernels/mamba/test_tree_wy_ref.py -k "not parity"
Run all (GPU):   pytest tests/kernels/mamba/test_tree_wy_ref.py
"""

import pytest
import torch

from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
    naive_tree_recurrence,
    tree_wy_batched,
    tree_wy_gated_delta_rule,
)


def _random_tree(n: int, gen: torch.Generator) -> torch.Tensor:
    """parent[t] in [0, t) for t>=1, root=-1 -> valid (parent[t] < t)."""
    parent = torch.full((n,), -1, dtype=torch.long)
    for t in range(1, n):
        parent[t] = torch.randint(0, t, (1,), generator=gen).item()
    return parent


def _make_inputs(n, h, kdim, vdim, gen, l2norm):
    q = torch.randn(n, h, kdim, generator=gen)
    k = torch.randn(n, h, kdim, generator=gen)
    v = torch.randn(n, h, vdim, generator=gen)
    beta = torch.rand(n, h, generator=gen)  # (0,1)
    # log-space decay: alpha = exp(g) in (0.5, 1) -> g in (log .5, 0)
    g = torch.log(0.5 + 0.5 * torch.rand(n, h, generator=gen))
    s0 = torch.randn(h, vdim, kdim, generator=gen)
    return q, k, v, g, beta, s0


# --------------------------------------------------------------------------
# CPU: tree-WY == naive recurrence (outputs + committed states), random trees
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(6)))
@pytest.mark.parametrize("h,kdim,vdim", [(1, 16, 16), (4, 16, 24), (2, 32, 32)])
def test_tree_wy_matches_naive(seed, h, kdim, vdim):
    gen = torch.Generator().manual_seed(1000 + seed)
    n = int(torch.randint(4, 40, (1,), generator=gen).item())
    parent = _random_tree(n, gen)
    q, k, v, g, beta, s0 = _make_inputs(n, h, kdim, vdim, gen, l2norm=True)

    out_naive, states = naive_tree_recurrence(q, k, v, g, beta, parent, s0)
    res = tree_wy_gated_delta_rule(q, k, v, g, beta, parent, s0)

    assert torch.allclose(res.output, out_naive, atol=1e-5, rtol=1e-4), (
        f"outputs differ: max {abs(res.output - out_naive).max():.2e}"
    )

    # reconstruct EVERY node's committed state from the single solve
    for a in range(n):
        sa = res.reconstruct_state(a)
        err = (sa - states[a]).abs().max()
        assert err < 1e-5, f"state[{a}] differs: max {err:.2e}"


def test_chain_is_special_case():
    """A chain (parent = path) must also match the naive recurrence -- this is
    the case chunk_gated_delta_rule covers; checks numerical parity."""
    gen = torch.Generator().manual_seed(7)
    n, h, kdim, vdim = 8, 2, 16, 16
    parent = torch.tensor([-1] + list(range(n - 1)), dtype=torch.long)  # 0<-1<-2...
    q, k, v, g, beta, s0 = _make_inputs(n, h, kdim, vdim, gen, l2norm=True)
    out_naive, states = naive_tree_recurrence(q, k, v, g, beta, parent, s0)
    res = tree_wy_gated_delta_rule(q, k, v, g, beta, parent, s0)
    assert torch.allclose(res.output, out_naive, atol=1e-5, rtol=1e-4)
    assert (res.reconstruct_state(n - 1) - states[n - 1]).abs().max() < 1e-5


# --------------------------------------------------------------------------
# CPU : BATCHED tree-WY over padded variable-size trees == naive
# --------------------------------------------------------------------------
@pytest.mark.parametrize("seed", list(range(4)))
def test_batched_tree_wy_matches_naive(seed):
    """tree_wy_batched shares ONE topology across the batch (that is what the GDN
    spec path has: every request verifies the same static draft tree), so the
    per-request rows must each match the naive recurrence for that shared tree."""
    gen = torch.Generator().manual_seed(500 + seed)
    bsz = int(torch.randint(2, 6, (1,), generator=gen).item())
    h, kdim, vdim = 3, 16, 24
    n = int(torch.randint(4, 30, (1,), generator=gen).item())
    parent = _random_tree(n, gen)  # [n] shared topology
    q = torch.randn(bsz, n, h, kdim, generator=gen)
    k = torch.randn(bsz, n, h, kdim, generator=gen)
    v = torch.randn(bsz, n, h, vdim, generator=gen)
    beta = torch.rand(bsz, n, h, generator=gen)
    g = torch.log(0.5 + 0.5 * torch.rand(bsz, n, h, generator=gen))
    s0 = torch.randn(bsz, h, vdim, kdim, generator=gen)
    accepted = torch.randint(0, n, (bsz,), generator=gen)

    res = tree_wy_batched(q, k, v, g, beta, parent, s0)
    states = res.reconstruct_states(accepted)
    for bi in range(bsz):
        out_n, st = naive_tree_recurrence(
            q[bi], k[bi], v[bi], g[bi], beta[bi], parent, s0[bi]
        )
        assert (res.output[bi] - out_n).abs().max() < 1e-5, f"row {bi} output"
        assert (states[bi] - st[int(accepted[bi])]).abs().max() < 1e-5, (
            f"row {bi} state"
        )


# --------------------------------------------------------------------------
# GPU : chain parity vs chunk_gated_delta_rule
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (B200)")
@pytest.mark.parametrize("h,kdim,vdim", [(2, 64, 64), (4, 128, 128)])
def test_parity_with_chunk_kernel(h, kdim, vdim):
    """On a chain, the tree-WY reference must match vLLM's chunk kernel.

    Knobs pinned to remove convention ambiguity: normalize q,k OUTSIDE,
    use_qk_l2norm_in_kernel=False, scale=1.0.  bf16 kernel -> loose tol.

    NOTE if this fails: first check whether the kernel decays the INITIAL state
    by the first token's gate (our reference does S_0 = alpha_0 S0 (I-..) + ..).
    """
    from vllm.third_party.flash_linear_attention.ops.chunk import chunk_gated_delta_rule
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import _l2norm

    dev = "cuda"
    n = 12
    gen = torch.Generator().manual_seed(3)
    q = _l2norm(torch.randn(n, h, kdim, generator=gen)).to(dev, torch.bfloat16)
    k = _l2norm(torch.randn(n, h, kdim, generator=gen)).to(dev, torch.bfloat16)
    v = torch.randn(n, h, vdim, generator=gen).to(dev, torch.bfloat16)
    beta = torch.rand(n, h, generator=gen).to(dev, torch.bfloat16)
    g = torch.log(0.5 + 0.5 * torch.rand(n, h, generator=gen)).to(dev, torch.float32)
    s0 = torch.randn(h, vdim, kdim, generator=gen).to(dev, torch.float32)

    # kernel: [B, T, H, *]; initial_state [N=B, H, V, K]
    out_k, final_state = chunk_gated_delta_rule(
        q=q.unsqueeze(0),
        k=k.unsqueeze(0),
        v=v.unsqueeze(0),
        g=g.unsqueeze(0),
        beta=beta.unsqueeze(0),
        scale=1.0,
        initial_state=s0.unsqueeze(0),
        output_final_state=True,
        use_qk_l2norm_in_kernel=False,
    )

    parent = torch.tensor([-1] + list(range(n - 1)), dtype=torch.long, device=dev)
    res = tree_wy_gated_delta_rule(
        q.float(),
        k.float(),
        v.float(),
        g,
        beta.float(),
        parent,
        s0,
        scale=1.0,
        use_qk_l2norm=False,
    )
    out_ref = res.output.to(out_k.dtype)  # [N, H, V]
    final_ref = res.reconstruct_state(n - 1)  # [H, V, K]

    torch.testing.assert_close(out_ref, out_k[0], atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        final_ref.to(final_state.dtype), final_state[0], atol=2e-2, rtol=2e-2
    )


# --------------------------------------------------------------------------
# D1 step-1 (GPU): tree-WY reconstruct == the ACTUAL spec rollback kernel
# (fused_sigmoid_gating_delta_rule_update) committed accepted state.
# This isolates the gate-convention risk (g = -exp(A_log)*softplus(a+dt_bias),
# beta = sigmoid(b), grouped heads 16k/32v, state layout [H,K,V]) BEFORE any
# hot-path edit. If it fails, suspects (in order): state transpose K<->V,
# head-group expansion direction, scale, l2norm.
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (B200)")
@pytest.mark.parametrize("num_spec", [1, 3])
def test_parity_with_fused_spec_kernel(num_spec):
    import torch.nn.functional as F

    from vllm.third_party.flash_linear_attention.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )

    dev = "cuda"
    dt = torch.bfloat16
    nk, nv, hd = 16, 32, 128  # Qwen3-Next: 16 key / 32 value heads, dim 128
    num_reqs = 4
    T = num_spec + 1  # tokens per req (chain)
    num_tokens = num_reqs * T
    total = num_tokens * 2
    torch.manual_seed(0)

    q = torch.rand(1, num_tokens, nk, hd, device=dev, dtype=dt)
    k = torch.rand(1, num_tokens, nk, hd, device=dev, dtype=dt)
    v = torch.rand(1, num_tokens, nv, hd, device=dev, dtype=dt)
    A_log = torch.rand(nv, device=dev, dtype=dt)
    dt_bias = torch.rand(nv, device=dev, dtype=dt)
    a = torch.rand(num_tokens, nv, device=dev, dtype=dt)
    b = torch.rand(num_tokens, nv, device=dev, dtype=dt)
    ssm_state = torch.rand(total, nv, hd, hd, device=dev, dtype=dt)  # [slot, H, K, V]
    state_indices = torch.randperm(total, device=dev, dtype=torch.int32)[
        :num_tokens
    ].view(num_reqs, T)
    num_accepted = torch.randint(1, T + 1, (num_reqs,), device=dev, dtype=torch.int32)
    cu_seqlens = torch.arange(0, num_tokens + 1, T, device=dev, dtype=torch.int32)

    # init state the kernel reads (accepted slot of each req), captured BEFORE the run
    rid = torch.arange(num_reqs, device=dev)
    init_slot = state_indices[rid, (num_accepted - 1).long()].long()
    s0 = ssm_state[init_slot].clone()  # [R, H, K, V]

    fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        initial_state=ssm_state,
        inplace_final_state=True,
        ssm_state_indices=state_indices,
        cu_seqlens=cu_seqlens,
        num_accepted_tokens=num_accepted,
        use_qk_l2norm_in_kernel=True,
    )
    committed = ssm_state[init_slot]  # [R, H, K, V] after

    # tree-WY reference: per-req chain, grouped k/q expanded 16->32 (v-head h uses
    # k-head h//2)
    qb = q.view(num_reqs, T, nk, hd).repeat_interleave(nv // nk, dim=2)
    kb = k.view(num_reqs, T, nk, hd).repeat_interleave(nv // nk, dim=2)
    vb = v.view(num_reqs, T, nv, hd)
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    gb = g.view(num_reqs, T, nv)
    betab = b.sigmoid().view(num_reqs, T, nv)
    parent = torch.tensor([-1] + list(range(T - 1)), device=dev)  # [T] shared chain
    # kernel state [slot, nv, hd, hd] is already [H, V, K] -> matches tree-WY S0,
    # with NO transpose.
    res = tree_wy_batched(qb, kb, vb, gb, betab, parent, s0, use_qk_l2norm=True)
    recon = res.reconstruct_states((num_accepted - 1).long())  # [R,H,V,K]

    torch.testing.assert_close(
        recon.to(committed.dtype), committed, atol=3e-2, rtol=3e-2
    )
