#!/usr/bin/env python3
"""CPU go/no-go for FIX A1 (two-K-source fused kernel) BEFORE any FA2 recompile.

A1 = one online-softmax pass reading paged CONTEXT (natural order) THEN the dense
SPINE-FIRST suffix, into a SINGLE fp32 accumulator, one bf16 round at the end (NO
merge_attn_states, NO intermediate bf16 -> no double-rounding, the flaw that killed
the dense-suffix hybrid).

This models the fp32 online-softmax faithfully (block-by-block rescale, fp32 accum,
final bf16 cast) and checks the two load-bearing properties:
  (1) A1-spine == spine-only  (the FIX: spine M-invariant, bit-exact)
  (2) A1 does NOT perturb the CONTEXT vs the fused single call (unlike the hybrid).
Controls: the SINGLE fused call and the double-rounding HYBRID, so we SEE A1 is
strictly better (hybrid perturbs context; A1 does not).

Data-independent arithmetic property -> synthetic random fp32 Q/K/V is a valid test.
"""
import torch

torch.manual_seed(0)
BF16 = torch.bfloat16
NEG = float("-inf")


def online_softmax(q, blocks, scale):
    """fp32 online-softmax. q:[M,D]; blocks: list of (K[Lb,D], V[Lb,D], bias[M,Lb] or None).
    Returns fp32 [M,D] (acc/l), matching the kernel's fp32 accumulation across blocks."""
    M, D = q.shape
    m = torch.full((M,), NEG, dtype=torch.float32)
    l = torch.zeros(M, dtype=torch.float32)
    acc = torch.zeros(M, D, dtype=torch.float32)
    for K, V, bias in blocks:
        s = (q @ K.t()) * scale
        if bias is not None:
            s = s + bias
        m_new = torch.maximum(m, s.max(dim=1).values)
        # rows that saw nothing yet + this block all -inf -> keep m=-inf safely
        m_safe = torch.where(torch.isfinite(m_new), m_new, torch.zeros_like(m_new))
        p = torch.exp(s - m_safe[:, None])
        p = torch.where(torch.isfinite(s), p, torch.zeros_like(p))
        alpha = torch.exp(torch.where(torch.isfinite(m), m, m_safe) - m_safe)
        alpha = torch.where(torch.isfinite(m), alpha, torch.zeros_like(alpha))
        l = l * alpha + p.sum(dim=1)
        acc = acc * alpha[:, None] + p @ V
        m = m_new
    return acc / l.clamp_min(1e-30)[:, None]


def main():
    C, tree_n, D = 512, 9, 64          # context keys, cat8 served nodes, head dim
    parent = [-1, 0, 0, 1, 1, 3, 3, 5, 7]   # cat8 served (root prepended)

    def anc(i):
        s = set()
        while i != -1:
            s.add(i); i = parent[i]
        return s

    spine = [0, 1, 3, 5, 7, 8]; branch = [2, 4, 6]; pi = spine + branch
    bias = torch.full((tree_n, tree_n), NEG, dtype=torch.float32)
    for i in range(tree_n):
        for j in anc(i):
            bias[i][j] = 0.0
    scale = 1.0 / (D ** 0.5)

    Q = torch.randn(tree_n, D, dtype=torch.float32)
    Kc = torch.randn(C, D, dtype=torch.float32); Vc = torch.randn(C, D, dtype=torch.float32)
    Ks = torch.randn(tree_n, D, dtype=torch.float32); Vs = torch.randn(tree_n, D, dtype=torch.float32)
    ctx_bias = torch.zeros(tree_n, C, dtype=torch.float32)   # every query sees all context

    def rb(x):  # round to bf16 then back to fp32 (the ONE final cast the kernel does)
        return x.to(BF16).to(torch.float32)

    # --- SINGLE fused call (reference): context block | suffix block (natural order) ---
    single = rb(online_softmax(Q, [(Kc, Vc, ctx_bias), (Ks, Vs, bias)], scale))

    # --- A1: context block | suffix SPINE-FIRST block, ONE fp32 accumulator ---
    pit = torch.tensor(pi)
    Ksp, Vsp, bsp = Ks[pit], Vs[pit], bias[:, pit]   # permute suffix cols spine-first
    a1_perm = online_softmax(Q, [(Kc, Vc, ctx_bias), (Ksp, Vsp, bsp)], scale)
    a1 = rb(a1_perm)  # rows still in natural query order (we permuted only KEY cols)

    # --- SPINE-ONLY (ground truth for the spine): context | spine-suffix, spine queries ---
    sp_idx = torch.tensor(spine)
    Qsp = Q[sp_idx]
    bias_sp = bias[sp_idx][:, sp_idx]
    spine_only = rb(online_softmax(Qsp, [(Kc, Vc, torch.zeros(len(spine), C)), (Ks[sp_idx], Vs[sp_idx], bias_sp)], scale))

    # --- HYBRID (double-rounding control): context->bf16, suffix->bf16, merge fp32 ---
    def split_lse(q, K, V, b):
        M, D2 = q.shape
        s = (q @ K.t()) * scale
        if b is not None:
            s = s + b
        mx = s.max(dim=1).values
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        p = torch.exp(s - mx[:, None])
        p = torch.where(torch.isfinite(s), p, torch.zeros_like(p))
        lse = mx + torch.log(p.sum(dim=1).clamp_min(1e-30))
        out = (p @ V) / p.sum(dim=1).clamp_min(1e-30)[:, None]
        return rb(out), lse   # <-- bf16 partial (double-round source)
    co, cl = split_lse(Q, Kc, Vc, ctx_bias)
    so, sl = split_lse(Q, Ksp, Vsp, bsp)
    w = torch.exp(cl - torch.maximum(cl, sl)); w2 = torch.exp(sl - torch.maximum(cl, sl))
    hybrid = rb((co * w[:, None] + so * w2[:, None]) / (w + w2)[:, None])

    def mx(a, b):
        return float((a - b).abs().max().item())

    print("=== FIX A1 CPU go/no-go (cat8: spine=%s branch=%s) ===" % (spine, branch))
    # (1) THE FIX: A1 spine rows == spine-only  (bit-exact?)
    a1_spine = a1[sp_idx]
    print("  (1) A1-spine vs spine-only        max_abs = %.3e   %s"
          % (mx(a1_spine, spine_only), "BIT-EXACT (FIX WORKS)" if mx(a1_spine, spine_only) == 0.0 else "NONZERO"))
    # single-spine vs spine-only = the bug we are fixing (should be ~1 ULP nonzero)
    print("      [control] single-spine vs spine-only max_abs = %.3e (the bug: interleaved != spine-only)"
          % mx(single[sp_idx], spine_only))
    # (2) CONTEXT LOSSLESSNESS: A1 vs single on BRANCH rows (context+own-suffix; A1 must NOT
    #     perturb context — only the suffix reorder should move branch rows, ~1 ULP)
    br_idx = torch.tensor(branch)
    print("  (2) A1 vs single, branch rows     max_abs = %.3e (reorder ~1 ULP; NOT gross)"
          % mx(a1[br_idx], single[br_idx]))
    print("      A1 vs single, ALL rows        max_abs = %.3e" % mx(a1, single))
    # (3) HYBRID control: does the double-rounding hybrid perturb vs single MORE than A1?
    print("  (3) [control] HYBRID vs single    max_abs = %.3e (double-round; A1 should be <= this)"
          % mx(hybrid, single))
    print("      [control] HYBRID-spine vs spine-only max_abs = %.3e" % mx(hybrid[sp_idx], spine_only))

    # (4) THE REAL SPLIT-BLOCK GATE (fp32-testable): the DEPLOYED single call folds the
    #     context-TAIL and the suffix in the SAME kBlockN block; A1 splits them into two
    #     blocks (context-tail | suffix). Even in fixed-order fp32 the RESCALE TIMING differs
    #     (one rescale vs two) -> is the CONTEXT/branch output bit-identical or ~1 ULP off?
    #     This is the exact "split-block context bit-identity" that killed the hybrid.
    kB = 128
    tail = C % kB if C % kB else kB          # last context block size
    Kc_head, Vc_head = Kc[:C - tail], Vc[:C - tail]
    Kc_tail, Vc_tail = Kc[C - tail:], Vc[C - tail:]
    ctxb_head = torch.zeros(tree_n, C - tail); ctxb_tail = torch.zeros(tree_n, tail)
    # context split into kB blocks
    head_blocks = [(Kc_head[i:i + kB], Vc_head[i:i + kB], ctxb_head[:, i:i + kB])
                   for i in range(0, C - tail, kB)]
    # DEPLOYED: ... | [context-tail + suffix-natural] in ONE block
    interleaved_last = (torch.cat([Kc_tail, Ks]), torch.cat([Vc_tail, Vs]),
                        torch.cat([ctxb_tail, bias], dim=1))
    single_real = rb(online_softmax(Q, head_blocks + [interleaved_last], scale))
    # A1: ... | [context-tail] | [suffix spine-first]  (two blocks)
    a1_real = rb(online_softmax(Q, head_blocks + [(Kc_tail, Vc_tail, ctxb_tail), (Ksp, Vsp, bsp)], scale))
    # DISAMBIGUATE: block-split WITHOUT reorder (suffix natural) -> isolates the split's perturbation
    a1_real_noreorder = rb(online_softmax(Q, head_blocks + [(Kc_tail, Vc_tail, ctxb_tail), (Ks, Vs, bias)], scale))
    print("  (4) SPLIT-BLOCK gate (context-tail interleaved vs split):")
    print("      [block-split ALONE, natural suffix] vs single_real, ALL rows = %.3e (0 => split lossless; nonzero => split PERTURBS)"
          % mx(a1_real_noreorder, single_real))
    print("      single_real vs A1_real, BRANCH rows = %.3e (context bit-identity; 0 => split is lossless)"
          % mx(a1_real[br_idx], single_real[br_idx]))
    print("      single_real vs A1_real, ALL rows     = %.3e" % mx(a1_real, single_real))
    split_clean = (mx(a1_real[br_idx], single_real[br_idx]) == 0.0)

    ok = (mx(a1_spine, spine_only) == 0.0)
    print(">>> GO/NO-GO (arithmetic, fixed-order): A1-spine==spine-only=%s ; double-round-free=%s ; "
          "split-block-context-lossless=%s" % (mx(a1_spine, spine_only) == 0.0,
          mx(a1, single) <= mx(hybrid, single), split_clean))
    print("    NOTE: fixed-order torch CANNOT reproduce the HW butterfly (single-spine-vs-spine-only=0 here");
    print("    is trivial). The BUTTERFLY fix is proven by the MAB REORDER arm on real HW. This CPU model")
    print("    proves (a) A1 avoids the hybrid double-round, (b) the split-block RESCALE is/ isn't fp32-lossless.")


if __name__ == "__main__":
    main()
