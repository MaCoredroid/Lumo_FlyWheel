# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parity test for the PRODUCTION tree-WY chain commit+forward Triton kernel
(tree_wy_chain_commit_capture_triton, used by reconstruct mamba_cache_mode) against the
validated torch reference (tree_wy_ref.reconstruct_chains_result).

The reference is itself parity-checked against the stock gated-delta kernel in
test_tree_wy_ref.py, so together these close the chain:
    production Triton kernel  ==  torch reference  ==  stock gated-delta kernel.

With a ZERO previous-step stash the commit is a mathematical no-op, which isolates
the WY forward (the transform + reading s0 from the slot-indexed ssm_state). GPU
only (the kernel is Triton).
"""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("num_spec", [1, 3])
@pytest.mark.parametrize("dot_bf16", [True, False])
def test_chain_capture_forward_matches_reference(num_spec, dot_bf16):
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
        reconstruct_chains_result,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
    )

    dev = "cuda"
    dt = torch.bfloat16
    nk, nv, hd = 16, 32, 128  # Qwen3.5: 16 key / 32 value heads, dim 128
    r = 4  # spec sequences
    N = num_spec + 1  # tokens per chain (uniform)
    B = r + 8  # state/stash pool (slots index into it)
    torch.manual_seed(0)

    qs = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    ks = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    vs = torch.randn(r * N, nv, hd, device=dev, dtype=dt)
    a = torch.randn(r * N, nv, device=dev, dtype=dt)
    b = torch.randn(r * N, nv, device=dev, dtype=dt)
    A_log = torch.randn(nv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nv, device=dev, dtype=torch.float32)

    # slot-indexed state pool [B, H, V, K]; each spec seq's committed block is a
    # distinct NONZERO slot (0 == NULL/padded -> no-op in the kernel).
    ssm_state = torch.randn(B, nv, hd, hd, device=dev, dtype=torch.float32)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    s0 = ssm_state[slots].clone()  # [r, H, V, K] init per seq

    # ZERO prev-step stash -> commit no-op -> pure forward from s0. leaf is then
    # irrelevant (it only selects the committed prefix, which is a no-op here).
    vt_stash = torch.zeros(B, nv, N, hd, device=dev, dtype=torch.float32)
    kk_stash = torch.zeros(B, N, nv, hd, device=dev, dtype=torch.float32)
    gc_stash = torch.zeros(B, nv, N, device=dev, dtype=torch.float32)
    leaf = torch.full((r,), N - 1, device=dev, dtype=torch.int64)

    out = tree_wy_chain_commit_capture_triton(
        qs,
        ks,
        vs,
        a,
        b,
        A_log,
        dt_bias,
        vt_stash,
        kk_stash,
        gc_stash,
        leaf,
        slots,
        ssm_state.clone(),  # kernel writes back in place; keep the input pristine
        N,
        None,
        dot_bf16=dot_bf16,
    )  # [r, N, nv, hd]

    ref = reconstruct_chains_result(
        qs,
        ks,
        vs,
        a,
        b,
        A_log,
        dt_bias,
        s0,
        N,
    ).output.reshape(r, N, nv, hd)

    tol = 5e-2 if dot_bf16 else 1e-2  # bf16 dot operands are looser
    torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("num_spec", [1, 3])
def test_chain_capture_commit_matches_reference(num_spec):
    """The COMMIT half, which the forward test above cannot see: it zeroes the
    stash, making the commit a mathematical no-op. Here the previous step's stash
    is NONZERO, so the kernel must reconstruct the accepted state
    exp(gcum_leaf)*s0 + (w*Vtilde)^T k  into the sequence's slot, and only then
    run the forward from it. Also asserts the stash the kernel writes back (next
    step's input) and that padded rows (slot 0 == NULL_BLOCK_ID) touch nothing.
    """
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
        reconstruct_chains_result,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
    )

    dev = "cuda"
    dt = torch.bfloat16
    nk, nv, hd = 16, 32, 128
    r, npad = 4, 2  # npad cudagraph-padded rows after r real ones
    N = num_spec + 1
    B = r + 8
    torch.manual_seed(0)

    rt = (r + npad) * N
    qs = torch.randn(rt, nk, hd, device=dev, dtype=dt)
    ks = torch.randn(rt, nk, hd, device=dev, dtype=dt)
    vs = torch.randn(rt, nv, hd, device=dev, dtype=dt)
    a = torch.randn(rt, nv, device=dev, dtype=torch.float32)
    b = torch.randn(rt, nv, device=dev, dtype=torch.float32)
    A_log = torch.randn(nv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nv, device=dev, dtype=torch.float32)

    ssm_state = torch.randn(B, nv, hd, hd, device=dev, dtype=torch.float32)
    slots = torch.cat(
        [
            torch.arange(1, r + 1, device=dev, dtype=torch.int64),
            torch.zeros(npad, device=dev, dtype=torch.int64),
        ]
    )
    leaf = torch.arange(r + npad, device=dev, dtype=torch.int64) % N

    # NONZERO prev-step stash. gcum is a cumulative LOG-decay, so it must be <= 0.
    vt_stash = torch.randn(B, nv, N, hd, device=dev, dtype=torch.float32)
    kk_stash = torch.randn(B, N, nv, hd, device=dev, dtype=torch.float32)
    gc_stash = -torch.rand(B, nv, N, device=dev, dtype=torch.float32) * 2.0

    s0_prev = ssm_state[slots[:r]].clone()
    vt_prev, kk_prev = vt_stash[slots[:r]].clone(), kk_stash[slots[:r]].clone()
    gc_prev = gc_stash[slots[:r]].clone()
    pristine = [t[0].clone() for t in (ssm_state, vt_stash, kk_stash, gc_stash)]

    out = tree_wy_chain_commit_capture_triton(
        qs,
        ks,
        vs,
        a,
        b,
        A_log,
        dt_bias,
        vt_stash,
        kk_stash,
        gc_stash,
        leaf,
        slots,
        ssm_state,
        N,
        None,
    )

    # reference commit: reconstruct_states() at the accepted leaf, from the PREV stash
    bi = torch.arange(r, device=dev)
    lf = leaf[:r]
    keep = (torch.arange(N, device=dev).unsqueeze(0) <= lf.unsqueeze(1)).float()
    gc_leaf = gc_prev[bi, :, lf]  # [r, nv]
    w = keep.unsqueeze(1) * torch.exp(
        (gc_leaf.unsqueeze(2) - gc_prev).clamp(max=0.0)
    )  # [r, nv, N]
    want = torch.exp(gc_leaf)[..., None, None] * s0_prev + torch.einsum(
        "bhi,bhiv,bhik->bhvk", w, vt_prev, kk_prev.permute(0, 2, 1, 3)
    )
    torch.testing.assert_close(ssm_state[slots[:r]], want, atol=0.5, rtol=5e-2)

    # the forward must then run from that freshly committed state, and the stash
    # written back must be the reference's Vtilde / l2-normed k for THIS step.
    ref = reconstruct_chains_result(
        qs[: r * N],
        ks[: r * N],
        vs[: r * N],
        a[: r * N],
        b[: r * N],
        A_log,
        dt_bias,
        want,
        N,
        None,
    )
    torch.testing.assert_close(
        out[:r].float(), ref.output.reshape(r, N, nv, hd).float(), atol=0.1, rtol=0.1
    )
    torch.testing.assert_close(vt_stash[slots[:r]], ref.vtilde, atol=0.1, rtol=0.1)
    torch.testing.assert_close(kk_stash[slots[:r]], ref._k, atol=5e-3, rtol=5e-3)

    # padded rows write nothing: the null block must be byte-identical.
    for got, exp in zip((ssm_state, vt_stash, kk_stash, gc_stash), pristine):
        assert torch.equal(got[0], exp), "padded row (slot 0) was written"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("num_spec", [1, 3])
def test_chain_capture_sentinel_skips_commit(num_spec):
    """A block whose stash belongs to a previously-finished sequence is marked by
    the prefill with a POSITIVE gcum sentinel (impossible for a real cumulative
    log-decay, which sums g <= 0). The mixer must then skip that one commit,
    leaving the committed state EXACTLY as it was -- the same result as committing
    from an all-zero stash (exp(0)*s0 + 0 == s0), which is why marking the small
    gcum tensor can replace zeroing the large Vtilde/k stashes.

    Guards the reconstruct prefill fast path: zeroing all three stash tensors per layer
    cost ~16% of prefill throughput (B200, in=4096, conc 512).
    """
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
        reconstruct_chains_result,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
    )

    dev = "cuda"
    dt = torch.bfloat16
    nk, nv, hd = 16, 32, 128
    r, N, B = 4, num_spec + 1, 12
    torch.manual_seed(0)

    qs = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    ks = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    vs = torch.randn(r * N, nv, hd, device=dev, dtype=dt)
    a = torch.randn(r * N, nv, device=dev, dtype=torch.float32)
    b = torch.randn(r * N, nv, device=dev, dtype=torch.float32)
    A_log = torch.randn(nv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nv, device=dev, dtype=torch.float32)

    ssm_state = torch.randn(B, nv, hd, hd, device=dev, dtype=torch.float32)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    leaf = torch.arange(r, device=dev, dtype=torch.int64) % N
    s0_prev = ssm_state[slots].clone()

    # Stale stash + the sentinel a new prefill writes. If the kernel wrongly
    # committed, this NONZERO stash would move the state well away from s0_prev.
    vt_stash = torch.randn(B, nv, N, hd, device=dev, dtype=torch.float32)
    kk_stash = torch.randn(B, N, nv, hd, device=dev, dtype=torch.float32)
    gc_stash = torch.full((B, nv, N), 1.0, device=dev, dtype=torch.float32)

    out = tree_wy_chain_commit_capture_triton(
        qs,
        ks,
        vs,
        a,
        b,
        A_log,
        dt_bias,
        vt_stash,
        kk_stash,
        gc_stash,
        leaf,
        slots,
        ssm_state,
        N,
        None,
    )

    # commit skipped => byte-identical state (no arithmetic ran on it at all)
    assert torch.equal(ssm_state[slots], s0_prev), (
        "sentinel-marked block was committed instead of skipped"
    )
    # and the forward must still be correct, running from that untouched state
    ref = reconstruct_chains_result(qs, ks, vs, a, b, A_log, dt_bias, s0_prev, N, None)
    torch.testing.assert_close(
        out.float(), ref.output.reshape(r, N, nv, hd).float(), atol=0.1, rtol=0.1
    )
    # the sentinel must self-clear: real gcum (<= 0) is written back this launch,
    # so the NEXT step commits normally.
    assert (gc_stash[slots] <= 0).all(), "sentinel not cleared by the mixer"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_chain_capture_smoke_shape_finite():
    """Cheap regression guard on the launcher signature + shapes: the kernel runs
    and returns finite output of shape [r, N, nv, hd]."""
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
    )

    dev = "cuda"
    dt = torch.bfloat16
    nk, nv, hd, r, N, B = 16, 32, 128, 3, 4, 11
    torch.manual_seed(0)
    qs = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    ks = torch.randn(r * N, nk, hd, device=dev, dtype=dt)
    vs = torch.randn(r * N, nv, hd, device=dev, dtype=dt)
    a = torch.randn(r * N, nv, device=dev, dtype=dt)
    b = torch.randn(r * N, nv, device=dev, dtype=dt)
    A_log = torch.randn(nv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(nv, device=dev, dtype=torch.float32)
    ssm_state = torch.randn(B, nv, hd, hd, device=dev, dtype=torch.float32)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    leaf = torch.full((r,), N - 1, device=dev, dtype=torch.int64)
    vt = torch.zeros(B, nv, N, hd, device=dev, dtype=torch.float32)
    kk = torch.zeros(B, N, nv, hd, device=dev, dtype=torch.float32)
    gc = torch.zeros(B, nv, N, device=dev, dtype=torch.float32)

    out = tree_wy_chain_commit_capture_triton(
        qs,
        ks,
        vs,
        a,
        b,
        A_log,
        dt_bias,
        vt,
        kk,
        gc,
        leaf,
        slots,
        ssm_state,
        N,
        None,
    )
    assert tuple(out.shape) == (r, N, nv, hd)
    assert torch.isfinite(out).all()
