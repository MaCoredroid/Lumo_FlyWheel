# FR13 — WY-batched vs sequential GDN verify: STRATEGIC VERDICT (workflow w1ah11lw2, 2026-06-08, source-verified)

## !! CORRECTION (user, 2026-06-09): WY is PARKED, NOT DEAD. This verdict recommended sequential because WY "can't be bit-exact" (abs-0.0, lines 7/15) — but the ACTUAL deliverable gate is per-depth ARGMAX + within-E5-floor (line 28 here already says "NOT max_abs"). WY fails abs-0.0 (different reduction tree, irreducible) but was NEVER disproven at the within-floor/argmax bar: commit 26c577a1 measured WY spine "ARGMAX-LOSSLESS 6/6, over-margin drift on irrelevant vocab"; the later 56%-reject (b8747d23, final-logit 3.32) was a WY-STATE drift that got a fix (8a975837: state 1.66e-3 -> 2.98e-8) and was abandoned (pivot 20be68a5) on the abs-0.0 argument BEFORE an e2e within-floor re-measure. So WY-at-the-within-floor-bar is OPEN, same revival logic that saved the sequential kernel from literal-0.0. **WE STAY ON THE SEQUENTIAL TREE-SCAN (bit-exact-by-construction = strictly safer + scales); WY is a banked fallback, NOT pursued unless the user explicitly says so.** Archived here for that reason.

## THE QUESTION (user): are WY-batched and native-sequential GDN mathematically equal, or can we get under the fp32 floor (bit-exact)?

## ANSWER
1. **ℝ-equal: YES, PROVEN.** WY/UT chunk form = exact algebraic identity to the sequential rank-1 recurrence (Schreiber–Van Loan WY; Yang et al. 2406.06484 NeurIPS'24; GDN 2412.06464 ICLR'25). Not an approximation. Our scan 7.45e-9 vs serial = fp noise around an exact identity.
2. **Under the fp32 floor (bit-exact to native): NO for the WY-batched kernel; YES only via a sequential rank-1 tree-scan.**

## WHY — the dispatch is VERIFIED IN LIVE SOURCE (the lynchpin)
- `gdn_linear_attn.py:1117` `if spec_sequence_masks is not None:` → `:1119` `fused_sigmoid_gating_delta_rule_update(...)` = the **SEQUENTIAL rank-1 recurrence** (`fused_sigmoid_gating.py:136` `for i_t in range(0,T)`, L158-167).
- `:1142` `if num_prefills > 0:` → `:1148` `chunk_gated_delta_rule` = **chunked-WY is PREFILL-ONLY, never the verify oracle.**
- `gdn_attn.py` `spec_sequence_masks` set from `num_decode_draft_tokens` ⇒ MTP-5 verify (the E5 baseline) runs the **sequential** kernel.
- Our kernel default = `use_wy: bool = False` (`fr10_gdn_tree_kernel.py:734/800`) — the sequential ancestor-replay path (L278-300) is already the default; WY is opt-in.

**Consequence:** the WY thesis ("align WY casts/order to native-chunked, no rewrite") is FALSIFIED — there is no chunked-WY incumbent at verify to align to. WY-batched materializes KKᵀ dots + a tiled triangular solve = a different summation tree than a serial rank-1 chain; ℝ-equal expressions in different op-orders cannot be forced bit-identical in fp32 (fp non-associativity). **Same impossibility class as the banked Triton→FA2 byte-exact result.** No cast/boundary tuning reconciles two different reduction trees.

## THE BIT-EXACT PATH: sequential rank-1 tree-scan (RECOMMENDED — option b)
Native's own kernel with the linear `i_t` token index replaced by a **tree-ancestry walk**. On the pure spine (MTP-5, no branches) it collapses to native's `T` loop **identically** — same op order, same fp32 reductions, same cast boundaries ⇒ bit-exact **by construction**. Off-spine branches = native sequential run on the node's path-to-root (theorem-backed distributional losslessness: SpecInfer Thm 4.2 / STree). Mechanically NOT from-scratch: our `use_wy=False` default IS this path; the work is to make it op-for-op identical to `fused_sigmoid_gating.py:152-168` (in-kernel l2norm +1e-6, q*scale, b_h*=exp(b_g), b_v-=sum(b_h*b_k), b_v*=sigmoid(b), b_h+=b_v⊗b_k, b_o=sum(b_h*b_q), all fp32 accumulate, b_g=-exp(A_log)*softplus(a+dt_bias)) + per-node ancestry masking + register-resident branch checkpointing. **Treat `_tree_gdn_wy_kernel` as oracle/throwaway, not the deliverable.**

## SPEED: fast enough (regime PROVEN; number needs live confirm)
Decode on GB10 is **weight-bandwidth-bound** (~273 GB/s LPDDR5x; ~27 GB weights/forward ≈ ~99 ms unavoidable). All-layer GDN state traffic ≈ ~150 MB ≈ ~0.55 ms (<1% of forward); scan FLOPs are ~3 orders under the roofline. A 5→10-node tree-scan ≈ doubles in-register scan work (µs/layer); weight stream unchanged ⇒ verify ~2× tokens per identical weight-load. **The +35.8% B=4 replay tax was HBM-traffic (state materialization/copies), NOT scan-FLOPs — avoidable by keeping per-node state in registers and writing only the committed accepted-path state to HBM.** RISK TO POLICE: if the tree-scan re-loads h0 or writes per-node intermediate state to HBM it re-creates the tax — a design constraint + a number to measure live.

## PROVEN vs NEEDS-LIVE
- PROVEN (code/math/cited): ℝ-equality; native verify = sequential (5 file:line verified); WY-batched can't be bit-exact (fp order); bandwidth-bound regime; branch losslessness is distributional (SpecInfer/Multi-Draft), not bit-exact.
- NEEDS LIVE (do NOT pre-declare): (i) sequential tree-scan hits per-layer **0.0 spine AND branches** at B=4 CUDA-captured; (ii) e2e within **E5 self-noise floor**; (iii) TPS / accept-per-event ≥ native + under the +35.8% tax with register-resident state.

## LITERATURE / BANNED-HACK
No off-the-shelf bit-exact lossless GDN tree-verify exists (STree 2505.14969 is diagonal-A only, can't cover GDN rank-1 Householder). No impossibility lower bound — bit-exactness is an engineering target (reduction order / batch-invariance). BANNED: routing the spine through native `fused_sigmoid_gating_delta_rule_update`/`causal_conv1d_update` as "our kernel" (splice=oracle only; gate runs splice-OFF, our kernel computing). Gate = per-depth argmax/distributional spine+branch, NOT max_abs; verdict = e2e vs E5.
