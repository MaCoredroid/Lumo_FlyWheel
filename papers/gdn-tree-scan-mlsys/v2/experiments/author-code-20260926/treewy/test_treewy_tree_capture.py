# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-cudagraph capture parity for the reconstruct TREE spec-decode commit+forward.

The tree path reuses the same fused Triton kernel as the chain path, feeding it
lifted tree ancestor masks and a verified leaf id instead of triangular masks and
a prefix length. These tests pin the three properties that makes that safe:

  1. the fused kernel's tree forward + branch commit match the validated torch
     reference (``tree_wy_ref.reconstruct_chains_result`` with the lifted parent),
  2. the same call is capturable into a CUDA graph and a replay picks up new
     leaf/slot buffer contents (i.e. nothing was baked in at capture time),
  3. the layer's init-built static tables reproduce what the eager, host-side
     builders computed, so no capture-blocking rebuild is needed at runtime.
"""

from typing import Any, cast

import pytest
import torch

from vllm.v1.spec_decode.tree_spec import (
    TreeTopology,
    ancestor_path_mask,
    build_ancestor_mask,
    verify_lifted_parent,
)

WIDTHS = [2, 2]  # 6 draft nodes -> 7 lifted verify slots


def _lifted_parent(widths):
    return verify_lifted_parent(TreeTopology(widths).parent).to(torch.int64)


def _tree_masks(parent_v, device):
    """The (anc_strict, anc_incl) pair the capture path feeds the kernel."""
    n = parent_v.shape[0]
    incl = build_ancestor_mask(parent_v)  # follows parent_v's device
    strict = incl & ~torch.eye(n, dtype=torch.bool, device=incl.device)
    return (
        strict.to(device=device, dtype=torch.float32),
        incl.to(device=device, dtype=torch.float32),
    )


def _inputs(r, n, device, seed=0):
    nk, nv, hd = 16, 32, 128  # Qwen3.5 head layout
    g = torch.Generator(device=device).manual_seed(seed)
    kw = dict(device=device, dtype=torch.bfloat16, generator=g)
    return dict(
        qs=torch.randn(r * n, nk, hd, **kw),
        ks=torch.randn(r * n, nk, hd, **kw),
        vs=torch.randn(r * n, nv, hd, **kw),
        a=torch.randn(r * n, nv, **kw),
        b=torch.randn(r * n, nv, **kw),
        A_log=torch.randn(nv, device=device, dtype=torch.float32, generator=g),
        dt_bias=torch.randn(nv, device=device, dtype=torch.float32, generator=g),
        nv=nv,
        hd=hd,
    )


def _zero_stash(pool, nv, n, hd, device):
    return (
        torch.zeros(pool, nv, n, hd, device=device, dtype=torch.float32),
        torch.zeros(pool, n, nv, hd, device=device, dtype=torch.float32),
        torch.zeros(pool, nv, n, device=device, dtype=torch.float32),
    )


# --------------------------------------------------------------------------
# 1. kernel (tree masks) == torch tree reference
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_tree_capture_forward_matches_reference():
    """With a zero prev-step stash the commit is a mathematical no-op, isolating
    the tree WY forward read from the slot-indexed ssm_state."""
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
        reconstruct_chains_result,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_tree_commit_capture_triton,
    )

    dev = "cuda"
    parent_v = _lifted_parent(WIDTHS)
    n = parent_v.shape[0]
    r, pool = 4, 12
    x = _inputs(r, n, dev)
    anc_s, anc_i = _tree_masks(parent_v, dev)

    ssm = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)  # 0 == NULL
    s0 = ssm[slots].clone()
    vt, kk, gc = _zero_stash(pool, x["nv"], n, x["hd"], dev)
    leaf = torch.full((r,), n - 1, device=dev, dtype=torch.int64)

    out = tree_wy_tree_commit_capture_triton(
        x["qs"],
        x["ks"],
        x["vs"],
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        anc_s,
        anc_i,
        vt,
        kk,
        gc,
        anc_i,
        leaf,
        slots,
        ssm.clone(),
        n,
        None,
        dot_bf16=True,
    )

    ref = reconstruct_chains_result(
        x["qs"],
        x["ks"],
        x["vs"],
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        s0,
        n,
        parent=parent_v.to(dev),
    ).output.reshape(r, n, x["nv"], x["hd"])

    torch.testing.assert_close(out.float(), ref.float(), atol=5e-2, rtol=5e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_tree_capture_commits_branch_leaf_state():
    """The commit must reconstruct the state of the accepted BRANCH leaf, not of a
    DFS prefix. Run the kernel once to populate the persistent stash, then run it
    again with a non-leftmost leaf and compare the committed ssm_state against the
    reference's reconstruct for that leaf."""
    from vllm.third_party.flash_linear_attention.ops.tree_wy_ref import (
        reconstruct_chains_result,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_tree_commit_capture_triton,
    )

    dev = "cuda"
    parent_v = _lifted_parent(WIDTHS).to(dev)
    n = parent_v.shape[0]
    r, pool = 3, 8
    x = _inputs(r, n, dev, seed=7)
    anc_s, anc_i = _tree_masks(parent_v, dev)

    ssm = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    s0_step1 = ssm[slots].clone()
    vt, kk, gc = _zero_stash(pool, x["nv"], n, x["hd"], dev)

    args = (x["qs"], x["ks"], x["vs"], x["a"], x["b"], x["A_log"], x["dt_bias"])
    # Step 1: zero stash -> commit is a no-op, but the step's (Vtilde, k, gcum)
    # land in the persistent stash for step 2's commit to consume.
    tree_wy_tree_commit_capture_triton(
        *args,
        anc_s,
        anc_i,
        vt,
        kk,
        gc,
        anc_i,
        torch.zeros(r, device=dev, dtype=torch.int64),
        slots,
        ssm,
        n,
        None,
        dot_bf16=False,
    )

    # Rightmost leaf of the 2x2 tree: a non-contiguous DFS ancestor path, so a
    # prefix commit would give a different answer.
    leaf_id = n - 1
    assert not bool(ancestor_path_mask(parent_v.cpu(), leaf_id)[: leaf_id + 1].all()), (
        "pick a leaf whose ancestor path is NOT a DFS prefix"
    )

    ssm_before = ssm.clone()
    tree_wy_tree_commit_capture_triton(
        *args,
        anc_s,
        anc_i,
        vt,
        kk,
        gc,
        anc_i,
        torch.full((r,), leaf_id, device=dev, dtype=torch.int64),
        slots,
        ssm,
        n,
        None,
        dot_bf16=False,
    )

    ref = reconstruct_chains_result(
        *args, s0_step1, n, parent=parent_v
    ).reconstruct_states(torch.full((r,), leaf_id, device=dev, dtype=torch.int64))
    torch.testing.assert_close(ssm[slots].float(), ref.float(), atol=2e-2, rtol=2e-2)
    # Sanity: the commit actually moved the state (guards a silent no-op pass).
    assert not torch.allclose(ssm[slots], ssm_before[slots], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_tree_capture_sentinel_skips_commit():
    """The mask-taking tree mixer must honour the prefill stash sentinel too.

    A block whose stash belongs to a previously-finished sequence is marked with a
    POSITIVE gcum (impossible for a real cumulative log-decay, which sums g <= 0);
    the mixer must skip that one commit and leave the committed state exactly as it
    was. Without this the tree path would read a stale stash after a prefill, since
    the layer no longer zeroes Vtilde/k -- see test_tree_wy_chain_capture.py for the
    chain counterpart and the throughput reason."""
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_tree_commit_capture_triton,
    )

    dev = "cuda"
    parent_v = _lifted_parent(WIDTHS).to(dev)
    n = parent_v.shape[0]
    r, pool = 4, 12
    x = _inputs(r, n, dev, seed=5)
    anc_s, anc_i = _tree_masks(parent_v, dev)

    ssm = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    s0_prev = ssm[slots].clone()
    leaf = torch.arange(r, device=dev, dtype=torch.int64) % n

    # Stale stash + the sentinel a new prefill writes. A wrongly-taken commit would
    # move the state far from s0_prev, because this stash is large and nonzero.
    vt = torch.randn(pool, x["nv"], n, x["hd"], device=dev)
    kk = torch.randn(pool, n, x["nv"], x["hd"], device=dev)
    gc = torch.full((pool, x["nv"], n), 1.0, device=dev)

    tree_wy_tree_commit_capture_triton(
        x["qs"],
        x["ks"],
        x["vs"],
        x["a"],
        x["b"],
        x["A_log"],
        x["dt_bias"],
        anc_s,
        anc_i,
        vt,
        kk,
        gc,
        anc_i,
        leaf,
        slots,
        ssm,
        n,
        None,
    )
    # commit skipped -> the committed base is untouched by the stale stash.
    torch.testing.assert_close(ssm[slots], s0_prev, atol=0, rtol=0)
    # and the sentinel self-clears: gcum now holds this step's real (<= 0) values.
    assert (gc[slots] <= 0).all(), "sentinel must be overwritten by the real gcum"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("num_spec", [1, 3, 6])
def test_maskfree_equals_masked_on_a_chain(num_spec):
    """The two mixers must agree wherever both are valid.

    The mask-free mixer bakes in the chain triangles; the mask-taking one reads
    them as tensors. Feeding the latter the explicit lower-triangular masks makes
    the two mathematically identical, so this pins the mask-free kernel (used for
    chain AND width-1 tree in production) against the mask-taking one the tree
    tests validate against the torch reference. Runs both over the same nonzero
    stash so the commit half is exercised, not just the forward."""
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
        tree_wy_tree_commit_capture_triton,
    )

    dev = "cuda"
    n = num_spec + 1
    r, pool = 4, 10
    x = _inputs(r, n, dev, seed=17)
    # A width-1 lifted tree parent IS a chain, so its ancestor masks are exactly
    # the triangles the mask-free kernel derives internally.
    parent_v = _lifted_parent([1] * num_spec)
    assert parent_v.shape[0] == n
    anc_s, anc_i = _tree_masks(parent_v, dev)

    ssm0 = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)
    vt0 = torch.randn(pool, x["nv"], n, x["hd"], device=dev)
    kk0 = torch.randn(pool, n, x["nv"], x["hd"], device=dev)
    # gcum must be <= 0 (a real cumulative log-decay) or the sentinel fires.
    gc0 = -torch.rand(pool, x["nv"], n, device=dev) * 2.0
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    leaf = torch.arange(r, device=dev, dtype=torch.int64) % n
    args = (x["qs"], x["ks"], x["vs"], x["a"], x["b"], x["A_log"], x["dt_bias"])

    def clones():
        return ssm0.clone(), vt0.clone(), kk0.clone(), gc0.clone()

    ssm_f, vt_f, kk_f, gc_f = clones()
    out_free = tree_wy_chain_commit_capture_triton(
        *args, vt_f, kk_f, gc_f, leaf, slots, ssm_f, n, None, dot_bf16=True
    )

    ssm_m, vt_m, kk_m, gc_m = clones()
    out_mask = tree_wy_tree_commit_capture_triton(
        *args,
        anc_s,
        anc_i,
        vt_m,
        kk_m,
        gc_m,
        anc_i,
        leaf,
        slots,
        ssm_m,
        n,
        None,
        dot_bf16=True,
    )

    torch.testing.assert_close(out_free.float(), out_mask.float(), atol=1e-2, rtol=1e-2)
    # the committed state and the stash written back for the next step must agree too
    torch.testing.assert_close(ssm_f, ssm_m, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(vt_f, vt_m, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(kk_f, kk_m, atol=0, rtol=0)
    torch.testing.assert_close(gc_f, gc_m, atol=0, rtol=0)


# --------------------------------------------------------------------------
# 2. the tree call is CUDA-graph capturable and re-reads its inputs on replay
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_tree_capture_replays_with_new_leaf():
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_tree_commit_capture_triton,
    )

    dev = "cuda"
    parent_v = _lifted_parent(WIDTHS).to(dev)
    n = parent_v.shape[0]
    r, pool = 4, 10
    x = _inputs(r, n, dev, seed=3)
    anc_s, anc_i = _tree_masks(parent_v, dev)
    ssm = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)
    vt, kk, gc = _zero_stash(pool, x["nv"], n, x["hd"], dev)
    # Persistent (graph-visible) per-request inputs, as the metadata builder keeps.
    slots = torch.arange(1, r + 1, device=dev, dtype=torch.int64)
    leaf = torch.zeros(r, device=dev, dtype=torch.int64)
    args = (x["qs"], x["ks"], x["vs"], x["a"], x["b"], x["A_log"], x["dt_bias"])

    def run():
        return tree_wy_tree_commit_capture_triton(
            *args,
            anc_s,
            anc_i,
            vt,
            kk,
            gc,
            anc_i,
            leaf,
            slots,
            ssm,
            n,
            None,
            dot_bf16=False,
        )

    run()  # warm up Triton autotune/compile outside the capture
    torch.accelerator.synchronize()

    # The kernel commits into ssm and rewrites the stash in place, so both are
    # step state: snapshot them and restore before each run being compared.
    state = [t.clone() for t in (ssm, vt, kk, gc)]

    def restore():
        for dst, src in zip((ssm, vt, kk, gc), state):
            dst.copy_(src)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    # Replay with a DIFFERENT accepted leaf written into the same buffer. If the
    # tree topology or the leaf had been baked in at capture time (a host-side
    # rebuild, a .nonzero(), a freshly allocated stash) this would either crash
    # or reproduce the capture-time answer.
    restore()
    leaf.fill_(n - 1)
    graph.replay()
    torch.accelerator.synchronize()
    replayed, ssm_replayed = captured.clone(), ssm.clone()

    restore()
    eager = run()
    torch.accelerator.synchronize()

    # Bit-exact: replay must run the same kernels on the same buffers.
    torch.testing.assert_close(replayed.float(), eager.float(), atol=0, rtol=0)
    torch.testing.assert_close(ssm_replayed, ssm, atol=0, rtol=0)
    # And the new leaf must actually have changed the answer vs the captured one,
    # otherwise the test would pass even if the graph ignored the leaf buffer.
    restore()
    leaf.fill_(0)
    baseline = run()
    torch.accelerator.synchronize()
    assert not torch.allclose(baseline.float(), eager.float(), atol=1e-3)


# --------------------------------------------------------------------------
# 3. the layer's own tree spec path captures, with padded NULL rows
# --------------------------------------------------------------------------
def _stub_layer(tree, device, nv, hd, n, pool):
    """A QwenGatedDeltaNetAttention carrying only what the reconstruct capture path
    reads, so the production method can be driven without building a model."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.third_party.flash_linear_attention.ops.tree_wy_triton import (
        tree_wy_chain_commit_capture_triton,
        tree_wy_tree_commit_capture_triton,
    )

    layer = object.__new__(QwenGatedDeltaNetAttention)
    layer._reconstruct_tree_static = tree
    layer._reconstruct_cap_bv = 32
    layer._reconstruct_cap_warps = 2
    layer._reconstruct_cap_stages = 3
    # Both mixer entry points, as __init__ binds them: _reconstruct_capture routes a
    # chain (or width-1 tree) to the mask-free one and a width>1 tree to the
    # mask-taking one, so the test exercises whichever production would pick.
    layer._tree_wy_chain_fn = tree_wy_chain_commit_capture_triton
    layer._tree_wy_tree_fn = tree_wy_tree_commit_capture_triton
    layer.activation = "silu"
    layer.A_log = torch.randn(nv, device=device)
    layer.dt_bias = torch.randn(nv, device=device)
    layer.kv_cache = (  # MambaBase declares this a tuple; slots 0/1 unused here
        None,
        None,
        torch.zeros(pool, nv, n, hd, device=device),
        torch.zeros(pool, n, nv, hd, device=device),
        torch.zeros(pool, nv, n, device=device),
    )
    return layer


class _Meta:
    """Stand-in for GDNAttentionMetadata: the capture path only reads
    accepted_leaf_ids and uses the object as a per-step scratch cache."""

    def __init__(self, accepted_leaf_ids):
        self.accepted_leaf_ids = accepted_leaf_ids


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
@pytest.mark.parametrize("widths", [WIDTHS, [1] * 6])
def test_layer_tree_spec_path_captures_with_padded_rows(widths):
    """_reconstruct_capture over a cudagraph-PADDED spec batch: real requests hold
    nonzero slots, padded rows hold NULL_BLOCK_ID and leaf -1. Capture must be
    bit-exact vs eager and must leave padded rows' state untouched.

    Both topologies matter: width>1 is the branching tree, and width-1 tree mode
    is the config that keeps FULL cudagraphs today (its verify needs no
    tree-specific softmax mask), so it is the one actually captured in serving.
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.spec_decode import tree_draft

    dev = "cuda"
    parent_v = _lifted_parent(widths)
    n = parent_v.shape[0]
    real, padded, pool = 3, 6, 12  # padded > real -> exercise the NULL rows
    x = _inputs(padded, n, dev, seed=11)

    tree_draft.set_tree_config(widths)
    try:
        tree = QwenGatedDeltaNetAttention._build_reconstruct_tree_static(
            cast(Any, object()), n, 4, dev
        )
    finally:
        tree_draft.set_tree_config(None)
    assert tree is not None
    assert tree["is_chain"] == all(w == 1 for w in widths)

    layer = _stub_layer(tree, dev, x["nv"], x["hd"], n, pool)
    ssm = torch.randn(pool, x["nv"], x["hd"], x["hd"], device=dev)

    # Persistent per-request metadata exactly as the builder pads it.
    spec_state_indices = torch.zeros(padded, 1, device=dev, dtype=torch.int32)
    spec_state_indices[:real, 0] = torch.arange(
        1, real + 1, device=dev, dtype=torch.int32
    )
    leaf_ids = torch.full((padded,), -1, device=dev, dtype=torch.int32)
    leaf_ids[:real] = 0
    num_accepted = torch.ones(padded, device=dev, dtype=torch.int32)

    def run():
        # A fresh metadata object each call, as the runner builds per step: the
        # capture path's dedup cache must never outlive a step.
        return layer._reconstruct_capture(
            x["qs"],
            x["ks"],
            x["vs"],
            x["a"],
            x["b"],
            ssm,
            spec_state_indices,
            num_accepted,
            _Meta(leaf_ids),
        )[0]

    run()
    torch.accelerator.synchronize()
    state = [t.clone() for t in (ssm, *layer.kv_cache[2:5])]

    def restore():
        for dst, src in zip((ssm, *layer.kv_cache[2:5]), state):
            dst.copy_(src)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    restore()
    leaf_ids[:real] = n - 2  # a deeper branch leaf, chosen at replay time
    graph.replay()
    torch.accelerator.synchronize()
    replayed, ssm_replayed = captured.clone(), ssm.clone()

    restore()
    eager = run()
    torch.accelerator.synchronize()

    torch.testing.assert_close(replayed.float(), eager.float(), atol=0, rtol=0)
    torch.testing.assert_close(ssm_replayed, ssm, atol=0, rtol=0)
    # Padded rows point at the null block; no real request's state may move
    # because of them.
    torch.testing.assert_close(ssm[real + 1 :], state[0][real + 1 :], atol=0, rtol=0)

    # The commit must be driven by accepted_leaf_ids, not by num_accepted_tokens:
    # num_accepted is constant here, so if the chain leaf were used the two leaf
    # settings would give identical committed state.
    restore()
    leaf_ids[:real] = 0
    run()
    torch.accelerator.synchronize()
    assert not torch.allclose(ssm[1 : real + 1], ssm_replayed[1 : real + 1], atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + Triton")
def test_layer_tree_conv_paths_capture():
    """The width>1 conv relocation and tree-ancestor conv output must also be
    capture-safe: static plans, padded shapes, no host sync."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.spec_decode import tree_draft

    dev = "cuda"
    parent_v = _lifted_parent(WIDTHS)
    n = parent_v.shape[0]
    k, channels, real, padded, pool = 4, 64, 2, 5, 9

    tree_draft.set_tree_config(WIDTHS)
    try:
        tree = QwenGatedDeltaNetAttention._build_reconstruct_tree_static(
            cast(Any, object()), n, k, dev
        )
    finally:
        tree_draft.set_tree_config(None)

    layer = _stub_layer(tree, dev, 1, 1, n, 1)
    conv_state = torch.randn(pool, channels, k - 1 + n, device=dev)
    x_raw = torch.randn(padded * n, channels, device=dev)
    conv_w = torch.randn(channels, k, device=dev)
    bias = torch.randn(channels, device=dev)
    spec_state_indices = torch.zeros(padded, 1, device=dev, dtype=torch.int32)
    spec_state_indices[:real, 0] = torch.arange(
        1, real + 1, device=dev, dtype=torch.int32
    )
    leaf_ids = torch.full((padded,), -1, device=dev, dtype=torch.int32)
    leaf_ids[:real] = n - 2
    num_accepted = torch.full((padded,), 2, device=dev, dtype=torch.int32)

    def run():
        meta = _Meta(leaf_ids)
        layer._reconstruct_compact_conv_state(conv_state, spec_state_indices, meta, n)
        return layer._reconstruct_ancestor_conv(
            x_raw, conv_state, spec_state_indices, num_accepted, conv_w, bias, n
        )

    conv_pristine = conv_state.clone()  # before any relocation has been applied
    run()
    torch.accelerator.synchronize()
    conv_at_capture = conv_state.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    conv_state.copy_(conv_at_capture)
    leaf_ids[:real] = 1  # different accepted branch on replay
    graph.replay()
    torch.accelerator.synchronize()
    replayed, conv_replayed = captured.clone(), conv_state.clone()

    conv_state.copy_(conv_at_capture)
    eager = run()
    torch.accelerator.synchronize()

    torch.testing.assert_close(replayed, eager, atol=0, rtol=0)
    torch.testing.assert_close(conv_replayed, conv_state, atol=0, rtol=0)
    # The null block (row 0) must be the only block padded rows can disturb.
    torch.testing.assert_close(
        conv_state[real + 1 :], conv_at_capture[real + 1 :], atol=0, rtol=0
    )
    # The relocation must follow accepted_leaf_ids: a different accepted branch
    # moves different conv columns. Assert on conv_state, not on the returned
    # output — every ancestor path starts at the anchor, so column base+0 never
    # moves and this step's conv window only reaches that far back. Start from
    # the PRISTINE buffer: relocation is idempotent, so re-applying the same
    # leaf's map to an already-relocated buffer is a no-op.
    conv_state.copy_(conv_pristine)
    leaf_ids[:real] = 0
    run()
    torch.accelerator.synchronize()
    conv_leftmost = conv_state.clone()

    conv_state.copy_(conv_pristine)
    leaf_ids[:real] = n - 2
    run()
    torch.accelerator.synchronize()
    assert not torch.allclose(
        conv_state[1 : real + 1], conv_leftmost[1 : real + 1], atol=1e-4
    )


# --------------------------------------------------------------------------
# 4. the shared topology stays on the HOST wherever it is first built
# --------------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_topology_is_cpu_pinned_under_cuda_default_device():
    """TreeTopology is host metadata and must not follow the ambient default device.

    vLLM constructs model layers inside `with torch.device("cuda")`. Because the
    GDN layer builds its static tree tables in __init__, that is where the shared
    topology cache now gets created first — so a bare torch.tensor() there would
    put `parent`/`node_depth` on the GPU and break every host consumer, e.g.
    gpu_model_runner._apply_tree_verify_mrope_positions' node_depth.numpy().
    """
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.spec_decode import tree_draft
    from vllm.v1.spec_decode.tree_spec import ancestor_path_mask

    with torch.device("cuda"):
        topo = TreeTopology(WIDTHS)
        assert topo.parent.device.type == "cpu"
        assert topo.node_depth.device.type == "cpu"
        assert all(t.device.type == "cpu" for t in topo.level_ids)
        assert all(t.device.type == "cpu" for t in topo.level_parent_local)
        # the host consumer that actually broke
        assert topo.node_depth.numpy().shape == (topo.num_nodes,)

        parent_v = verify_lifted_parent(topo.parent)
        assert parent_v.device.type == "cpu"
        # these two follow their input's device, so they stay on the host too
        assert build_ancestor_mask(parent_v).device.type == "cpu"
        assert ancestor_path_mask(parent_v, 1).device.type == "cpu"

        # and the layer's own init-time build must work in this context, landing
        # its tables on the requested device.
        tree_draft.set_tree_config(WIDTHS)
        try:
            tree = QwenGatedDeltaNetAttention._build_reconstruct_tree_static(
                cast(Any, object()), topo.num_nodes + 1, 4, "cuda"
            )
        finally:
            tree_draft.set_tree_config(None)
    assert tree is not None
    for key in ("anc_s", "anc_i", "reloc", "conv_ti", "conv_hi", "parent_v"):
        assert tree[key].device.type == "cuda", key


# --------------------------------------------------------------------------
# 5. init-built static tables == the eager host-side builders
# --------------------------------------------------------------------------
@pytest.mark.parametrize("widths", [[1, 1, 1], [2, 2], [3, 2]])
def test_static_tables_match_eager_builders(widths, monkeypatch):
    """The layer precomputes these at __init__ precisely because the eager
    versions need .nonzero()/.tolist(); this pins them to the same values."""
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    from vllm.v1.spec_decode import tree_draft

    tree_draft.set_tree_config(widths)
    monkeypatch.setattr(tree_draft, "_WIDTHS", list(widths))

    parent_v = _lifted_parent(widths)
    n = parent_v.shape[0]
    k = 4
    # Unbound call: the builder only reads the tree topology and its arguments,
    # never layer state, so it needs no constructed layer.
    tree = QwenGatedDeltaNetAttention._build_reconstruct_tree_static(
        cast(Any, object()), n, k, "cpu"
    )

    assert tree is not None, "tree drafting must be enabled for this test"
    assert tree["n"] == n and tree["k"] == k
    assert tree["is_chain"] == all(w == 1 for w in widths)
    assert torch.equal(tree["parent_v"], parent_v)

    incl, strict = _tree_masks(parent_v, "cpu")[1], _tree_masks(parent_v, "cpu")[0]
    torch.testing.assert_close(tree["anc_i"], incl)
    torch.testing.assert_close(tree["anc_s"], strict)
    # anc_i doubles as the commit keep-table: row L == ancestor path of leaf L.
    for leaf in range(n):
        torch.testing.assert_close(
            tree["anc_i"][leaf],
            ancestor_path_mask(parent_v, leaf).to(torch.float32),
        )

    # reloc: row L maps the accepted path of L onto the contiguous prefix; row 0
    # and every column past the path end stays the identity.
    for leaf in range(n):
        path = sorted(
            int(i) for i in ancestor_path_mask(parent_v, leaf).nonzero().flatten()
        )
        expected = list(range(n))
        expected[: len(path)] = path
        assert tree["reloc"][leaf].tolist() == expected, leaf

    # conv plan: exactly one source per (slot, window position), the current token
    # last, and successively older positions walking the lifted parent chain.
    ti, hi = tree["conv_ti"], tree["conv_hi"]
    assert ((ti >= 0) ^ (hi >= 0)).all()
    assert torch.equal(ti[:, k - 1], torch.arange(n))
    for s in range(n):
        cur = s
        for kk in range(k - 2, -1, -1):
            cur = int(parent_v[cur]) if cur >= 0 else -1
            if cur >= 0:
                assert int(ti[s, kk]) == cur
            else:
                assert int(hi[s, kk]) >= 0
