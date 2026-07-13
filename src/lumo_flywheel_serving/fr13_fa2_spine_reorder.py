"""FR13_FA2_SPINE_REORDER — live dense-suffix hybrid for FIX A (contiguous-spine reorder).

Fixes the cat8 spine accept-rate M-dependence: the forked FA2 tree-verify decode
gives the deep-spine row a 1-bf16-ULP error (6.25e-2) because interleaved BRANCH
tree nodes place the surviving SPINE keys at different score-tile columns ->
different online-softmax butterfly-reduction association. Validated fix (MAB, whole
spine bit-exact vs spine-only): permute the tree suffix so SPINE nodes are
contiguous-first, pi = [spine depth-order] ++ [branches topological].

The LIVE kernel reads PAGED KV (block_table), and the tree-node cache slots are
also read positionally by the GDN + committer (carrier-B). So instead of touching
the cache layout, we do the vLLM cascade/DCP split at the FA2 call ONLY:
  (1) CONTEXT: paged, causal=False, tree_bias=None, seqused_k = seq_lens - tree_n
      (drops the just-written tree suffix), return LSE.
  (2) SUFFIX: DENSE over the current-step key/value tree rows, permuted spine-first
      (q/k/v + both tree_bias axes), causal=True + permuted bias, return LSE.
  (3) merge_attn_states(output, ctx, ctx_lse, suffix_unpermuted, suffix_lse_unpermuted).
The permutation lives ONLY in local temporaries + is inverted before merge into the
flat `output`, so the cache / slot_mapping / committer / GDN node order are all
UNTOUCHED (zero carrier-B consumers move). Lossless by exact relabel; no-HBM-tax
(context read once as now; suffix is <=16 tokens from the in-hand key/value).

This module is PURE (functions take the flash + merge ops as params) so its
permute / un-permute / merge-order logic is CPU-testable with stubs before it is
wired into the live tree_attn patch.
"""
from __future__ import annotations

import sys

import torch

_ENGAGED_ONCE = [False]
_SELFCHECK_N = [0]        # count of self-check forwards logged
_SELFCHECK_MAX = [0.0]    # running max over all self-check forwards


def _mark_engaged(mode: str) -> None:
    """One-time loud stderr marker so the boot log PROVES the hybrid ran (guards
    against the patch emitting the single-call fallback because the anchor didn't
    match the container's tree_attn.py version -- a silent no-op)."""
    if not _ENGAGED_ONCE[0]:
        _ENGAGED_ONCE[0] = True
        try:
            print(
                "FR13_FA2_SPINE_REORDER hybrid ENGAGED (mode=%s)" % mode,
                file=sys.stderr, flush=True,
            )
        except Exception:
            pass


def spine_first_perm(bias_2d: torch.Tensor):
    """Derive the spine-first permutation pi (+ inverse) from a [tree_n, tree_n] tree_bias.

    The DEEPEST tree node's visible set (finite bias entries in its row) is the
    all-zeros spine (longest root->leaf path); pi = that spine (ascending flat
    index == depth order for a caterpillar) followed by the remaining BRANCH nodes
    in ascending flat index (topological: a parent's flat index < its child's).
    Threshold-free visible test (finite & > -1e30 handles -inf and finfo.min masks).

    Returns (pi, inv) as long tensors on bias_2d.device, or (None, None) if the tree
    is degenerate (<2 nodes) or already fully contiguous-spine (pi == identity).
    """
    n = int(bias_2d.shape[-1])
    if n < 2:
        return None, None
    sub = bias_2d[:n, :n]
    vis = torch.isfinite(sub) & (sub > -1e30)
    deep = int(vis.sum(dim=1).argmax().item())
    spine = [j for j in range(n) if bool(vis[deep, j].item())]
    if len(spine) < 2:
        return None, None
    spine_set = set(spine)
    branch = [j for j in range(n) if j not in spine_set]
    pi_list = spine + branch  # spine ascending (depth order) then branches ascending
    if pi_list == list(range(n)):
        return None, None  # already contiguous-spine (e.g. a linear chain) -> no-op
    pi = torch.tensor(pi_list, dtype=torch.long, device=bias_2d.device)
    inv = torch.empty_like(pi)
    inv[pi] = torch.arange(n, device=bias_2d.device)
    return pi, inv


def _permute_bias(bias_2d: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Permute BOTH axes of a [.., n, n] tree_bias by pi (rows=query, cols=key)."""
    return bias_2d.index_select(-2, pi).index_select(-1, pi).contiguous()


def hybrid_reorder_decode(
    *,
    query,          # [num_decode, H, D]  (per-request tree query rows, flat order)
    key,            # [num_decode, Hkv, D] fresh current-step K (flat order)
    value,          # [num_decode, Hkv, D] fresh current-step V (flat order)
    key_cache,
    value_cache,
    output,         # [num_decode, H, D] (or [num_decode, H*D]); written in place
    cu_seqlens_q,   # [B+1] per-request query boundaries
    seq_lens,       # [B] per-request total KV length (context + tree_n)
    max_query_len,  # == tree_n (homogeneous trees)
    max_seq_len,
    block_table,
    tree_bias,      # [tree_n, tree_n] or [B, tree_n, tree_n]
    scale,
    softcap,
    sliding_window_size,
    flash_fn,       # flash_attn_varlen_func
    merge_fn,       # merge_attn_states
    fa_version=2,
    split_only=False,   # gate-1a: run the cascade split with pi=identity (NO reorder)
                        # to prove context+suffix+merge == the single paged call.
    self_check=False,   # =3 debug: split->temp + single-ref->output; log the diff.
):
    """Dense-suffix hybrid decode with spine-first suffix reorder. Homogeneous trees
    (every request has the same tree_n == max_query_len). Returns True if the reorder
    path ran, False if it declined (caller should fall back to the single call)."""
    tree_n = int(max_query_len)
    B = int(cu_seqlens_q.shape[0]) - 1
    if tree_n < 2 or B < 1 or query.shape[0] != B * tree_n:
        return False  # non-homogeneous / not the tree-decode shape -> caller falls back
    bias0 = tree_bias[0] if tree_bias.dim() == 3 else tree_bias
    if split_only:
        pi = torch.arange(tree_n, dtype=torch.long, device=query.device)
        inv = pi
    else:
        pi, inv = spine_first_perm(bias0)
        if pi is None:
            return False  # nothing to reorder (already contiguous / degenerate)
    _mark_engaged("split_only" if split_only else "reorder")
    dev = query.device
    H = query.shape[1]
    out3 = output.view(B * tree_n, H, -1)  # ensure [N, H, D] view for merge

    # (1) CONTEXT — paged, non-causal, NO tree bias, suffix excluded.
    ctx_out = torch.empty_like(out3)
    ctx_out, ctx_lse = flash_fn(
        q=query, k=key_cache, v=value_cache, out=ctx_out,
        cu_seqlens_q=cu_seqlens_q, max_seqlen_q=tree_n,
        seqused_k=seq_lens - tree_n, max_seqlen_k=max_seq_len,
        softmax_scale=scale, causal=False, window_size=sliding_window_size,
        block_table=block_table, softcap=softcap,
        return_softmax_lse=True, fa_version=fa_version, tree_bias=None,
    )

    # (2) SUFFIX — dense, permuted spine-first (per request), causal + permuted bias.
    q3 = query.view(B, tree_n, H, -1)
    k3 = key.view(B, tree_n, key.shape[1], -1)
    v3 = value.view(B, tree_n, value.shape[1], -1)
    qp = q3.index_select(1, pi).reshape(B * tree_n, H, -1).contiguous()
    kp = k3.index_select(1, pi).reshape(B * tree_n, key.shape[1], -1).contiguous()
    vp = v3.index_select(1, pi).reshape(B * tree_n, value.shape[1], -1).contiguous()
    tb_p = _permute_bias(bias0, pi)
    if tree_bias.dim() == 3:
        tb_p = tb_p.unsqueeze(0).expand(B, tree_n, tree_n).contiguous()
    cu_tree = torch.arange(0, (B + 1) * tree_n, tree_n, dtype=torch.int32, device=dev)
    suf_out = torch.empty_like(qp)
    suf_out, suf_lse = flash_fn(
        q=qp, k=kp, v=vp, out=suf_out,
        cu_seqlens_q=cu_tree, max_seqlen_q=tree_n,
        cu_seqlens_k=cu_tree, max_seqlen_k=tree_n,
        softmax_scale=scale, causal=True, window_size=sliding_window_size,
        softcap=softcap, return_softmax_lse=True, fa_version=fa_version, tree_bias=tb_p,
    )

    # (3) un-permute suffix rows + LSE back to flat node order, then merge.
    suf_out_u = torch.empty_like(suf_out)
    suf_out_v = suf_out.view(B, tree_n, H, -1)
    suf_out_u.view(B, tree_n, H, -1).index_copy_(1, pi, suf_out_v)
    # suf_lse is [H, B*tree_n]; permute the token axis per request.
    suf_lse_u = torch.empty_like(suf_lse)
    slu = suf_lse_u.view(suf_lse.shape[0], B, tree_n)
    slv = suf_lse.view(suf_lse.shape[0], B, tree_n)
    slu.index_copy_(2, pi, slv)

    if self_check:
        # BISECT: split -> temp; SINGLE reference (paged, full, causal, tree_bias)
        # -> out3 (the SAFE live output).  Log max_abs(split - single) + component
        # diagnostics ONCE so a broken split is localized without shipping it.
        merged = torch.empty_like(out3)
        merge_fn(merged, ctx_out, ctx_lse, suf_out_u, suf_lse_u)
        flash_fn(
            q=query, k=key_cache, v=value_cache, out=out3,
            cu_seqlens_q=cu_seqlens_q, max_seqlen_q=tree_n,
            seqused_k=seq_lens, max_seqlen_k=max_seq_len,
            softmax_scale=scale, causal=True, window_size=sliding_window_size,
            block_table=block_table, softcap=softcap, fa_version=fa_version,
            tree_bias=tree_bias,
        )
        if _SELFCHECK_N[0] < 30:
            try:
                d = float((merged - out3).abs().max().item())
                _SELFCHECK_MAX[0] = max(_SELFCHECK_MAX[0], d)
                _SELFCHECK_N[0] += 1
                print(
                    "FR13_FA2_SPINE_REORDER SELFCHECK[%d] max_abs(split-single)=%.4e "
                    "RUNMAX=%.4e single_absmax=%.3e ctx_lse[%.2f,%.2f] suf_lse[%.2f,%.2f]"
                    % (_SELFCHECK_N[0], d, _SELFCHECK_MAX[0], out3.abs().max().item(),
                       ctx_lse.min().item(), ctx_lse.max().item(),
                       suf_lse.min().item(), suf_lse.max().item()),
                    file=sys.stderr, flush=True,
                )
            except Exception as _e:
                print("SELFCHECK log failed: %r" % _e, file=sys.stderr, flush=True)
        return True

    merge_fn(out3, ctx_out, ctx_lse, suf_out_u, suf_lse_u)
    return True
