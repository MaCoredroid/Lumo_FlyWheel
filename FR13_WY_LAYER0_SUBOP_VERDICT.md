# FR13 WY layer-0 sub-op localization — VERDICT (workflow wey8cg5w6, 2026-06-08, grounded)

## ONE LINE: layer-0 0.015625 = STANDING NUMERIC fp-seam in the WY scan recurrence (op-order + missing beta-bf16 cast). NOT a WY regression, NOT structural/mask. Fix = make OUR kernel bit-exact to native's recurrence; do NOT splice native.

## ROOT (ranked)
**#1 WY scan recurrence op-order (OURS).** `fr10_gdn_tree_kernel.py` `_tree_gdn_wy_kernel`: KKT build L527-540, UT/WY triangular solve L549-599, state/output recurrence L612-699. Native decode-regime ref = `/tmp/vllm_live_019/.../fla/ops/fused_recurrent.py` L326-329 = a **strictly sequential rank-1 update** (`b_h*=exp(g); b_v-=sum(b_h*b_k); b_v*=beta; b_h+=b_v*b_k`). Our WY computes the algebraically-equivalent answer via batched `tl.dot` Gram + triangular solve — a **different fp32 reduction order** → ~1% rel drift = 2 bf16 ULP (0.015625) on the single |value|≈1.5 channel (h31/d26/flat-3994).
**#2 beta not bf16-rounded (OURS).** L491 `b_beta = tl.sigmoid(b_b)` stays fp32; native L324 rounds beta through bf16 BEFORE the recurrence. beta multiplies every delta (L528/L534/L558-559/L674) — a real per-step seam. (q/k bf16-tapped L510-511; v/y L563/586/604; beta is the conspicuous gap.)

## REGRESSION verdict: STANDING (not a regression)
- Commit 62516997 = "FR12: verify L0 branch path gate" = diagnostic/capture-only (no kernel numerics). The MEMORY "0.0" = per-subkernel gates on CLEAN OFFLINE-REPLAY inputs, NOT the live composite layer-0 hidden.
- OLD ancestor-replay kernel read the SAME live 0.015625 on the same spine tokens (`FR12_PARITY_RESULTS.md`). WY replaced a same-magnitude seam (replay-conv) with a same-magnitude seam (WY-recurrence). **No 0.0 baseline to restore — the target is native bit-exact, never achieved live.**

## STRUCTURAL vs NUMERIC: NUMERIC (kernel fix), decisively
- abs-outliers (h31/d26/flat-3994) = the only |value|>0.4 channel = magnitude artifact (1-2 ULP, *tighter* than the layer median rel_ulp 3.0/mean 14.5).
- Error is BROAD not sparse: 11% bit-exact, 89% off ≥1 ULP, ~47% within 2 ULP — opposite of a structural subset seam.
- Per-row rel error uniform [14.1,10.9,8.9,18.6,21.0,13.9] — no branch/deep-row/per-head concentration. diff scales with |native activation| = fp32-order + bf16-boundary signature.

## NEXT GPU STEP (codex, minimal): layer-0 per-sub-op paired ladder
One paired run, B=1 eager, pinned prompt "Explain hash tables." (reuse the paired harness + native-on-path oracle, no new oracle). Dump layer-0 ONLY, tree-spine [0,1,2,4,6,8] ↔ native [0..5], a per-sub-op dump INSIDE forward_cuda (NOT another readout tap):
- in_proj (fp8, shared) → expect 0.0 (re-confirm fp8 batch-inv at B=1 live)
- causal_conv1d (OURS) → expect 0.0 LIVE (offline clean; FR12 once pinned 0.125 here — must re-confirm on LIVE input)
- l2norm(q,k) (OURS) → ~0.0
- **WY scan state-store (L705-708 vs native ht L334-335) → first nonzero expected**
- **WY scan output (L700-704 vs native b_o L330-331) → first nonzero expected (the 0.015625)**
- gate/o_proj (shared) → inherit scan drift only
SPLIT state-store diff vs output diff: state clean + output diverges ⇒ readout tl.dot order (L677/L696-697); state diverges ⇒ recurrence pass (L612-694).

## FIX (bit-exact-or-bust, after capture confirms scan-first)
(a) add beta bf16 cast at L491 to match native L324; (b) align the WY recurrence fp32 accumulation to native's sequential rank-1 order so bf16 boundaries land identically. Drive layer-0 scan output → 0.0, re-run the 64-layer ladder. **STRATEGIC NOTE:** native's verify-regime GDN is the SEQUENTIAL recurrent kernel (fused_recurrent), our WY is a batched solve. If the batched order can't be reconciled, the bit-exact path is a **sequential rank-1 tree-scan** (still OUR kernel, one-pass, shared state + ancestry mask; plausibly fast — only ~10 rank-1 updates/layer) — NOT the WY solve. Decide after the op-order grind.

## REWARD-HACK FLAG: NONE for the legit fix. BANNED = routing the spine through native fused_recurrent/causal_conv1d_update to pass the metric (splice = oracle ONLY). The capture uses native-on-path purely as the comparison oracle (sanctioned). No self-declare; the live sub-op ladder is the gate.
