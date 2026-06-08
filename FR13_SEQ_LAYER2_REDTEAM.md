# FR13 seq tree-scan layer-2 — monitor RED-TEAM of ww7rx446u (2026-06-08): localization SOLID, beta-bf16 fix WRONG

## VERIFIED AT LIVE SOURCE: do NOT add a bf16 cast to beta (it injects drift)
- Verify oracle = `fused_sigmoid_gating.py` (recurrence loop DIRECTLY at L136, NOT a call to fused_recurrent). Beta at **L150**: `b_beta = tl.sigmoid(b_b.to(tl.float32))` = **PURE fp32, no bf16**.
- OUR kernel **L332**: `b_beta = tl.sigmoid(b_b.to(tl.float32))` = **pure fp32 → ALREADY MATCHES the oracle.**
- `fused_recurrent.py:324`: `beta_val = tl.sigmoid(b_val).to(b.dtype.element_ty).to(tl.float32)` = bf16-rounded — but this is a **DIFFERENT, NON-verify kernel** (verify dispatch = fused_sigmoid_gating per w1ah11lw2 / gdn_linear_attn.py:1117). ww7rx446u + the old WY verdict cited the WRONG native kernel. w4abw0spa already corrected this; ww7rx446u re-introduced the error.
- **CONCLUSION: ww7rx446u fix #1 (bf16 cast on beta L332) is WRONG — it would move our beta AWAY from the verify oracle. DO NOT apply it. Same for any bf16-tap on the seq scan ops that the oracle does in fp32.**

## What IS solid from ww7rx446u (the localization, keep)
- Layers 0,1 bit-exact 0.0 (scan thesis proven). Layer-2 first divergence is confined to **spine-row 2 (pos 7)**, the first tree node past the first branch point.
- The outlier is **exactly 1 bf16 ULP**: tree `-2.18750` vs native `-2.203125` (|val|≈2.2 → ULP 2^-6 = 0.015625), at **flat-chan 3994 = head 31, head_dim 26** (model-dim/o_proj space) — the SAME high-magnitude channel as the WY layer-0 outlier (migrated up via state: seq fixed L0/L1, so the first place h31/d26 accumulates enough magnitude to straddle a bf16 bucket is now depth-2).
- Ruled out (per-row evidence): h0-misindex (single shared bank, rows 0/1/3/4/5 bit-exact), conv (row-2 windows identical [2,3,4,5]==native), fp8 (batch-invariant B=1), data-outlier (fp-rounding signature, not structural).

## THE REAL SOURCE: unknown — find it with the LIVE sub-op ladder (NOT beta)
Since beta already matches and w4abw0spa verified E1-E12 op-for-op, the 1-ULP at row-2/depth-2 must be a subtle fp32/cast/state-handling difference our op-for-op audit didn't catch — candidate: the **tree-specific parent-resume / h_cache path** (L277-283, our `tl.where`/`tl.sum` state extraction vs native's direct register carry of b_h) at the first high-magnitude depth, or a readout/state cast boundary. The live sub-op ladder localizes it definitively.

## DIRECTIVE TO CODEX
1. Do NOT add bf16 to beta/the seq scan (verified wrong above).
2. FIX the sub-op capture tooling: the native L2 sub-op capture returned HTTP 200 but wrote NO files (FR12 hooks didn't fire despite env present) — debug the hook wiring (why no write), get it producing per-sub-op .pt on row 2.
3. Run the LIVE layer-2 sub-op ladder, same L2 input both arms, row-2 specifically: in_proj(0.0 expected) -> post-conv(0.0 expected, windows identical) -> l2norm/g/beta into scan -> **scan-out (find if this is the 1-ULP injector)** -> RMSNormGated -> o_proj. The FIRST sub-op with 2^-6 divergence on identical row-2 input is the injector. If scan-out is first nonzero, inspect the h_cache parent-resume + the rank-1/readout casts (NOT beta).
4. Bind to FR13_LADDER_LOG.md; ONE GPU; no splice; no self-declare.
