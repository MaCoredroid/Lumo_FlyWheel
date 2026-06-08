# FR13 sequential tree-scan — pre-verified prep (workflow w4abw0spa, 2026-06-08, source-verified)

## HEADLINE: the rewrite is DONE (HEAD 88212830) + the math is ALREADY op-for-op bit-exact to native. ZERO kernel edits for bit-exactness. Two real jobs remain: (E6) verify a scale wiring, and (G1/G2) close two HBM-leak items for the TPS de-risk. Then run the live 0.0 ladder.

## CORRECTION to the task premise + my pivot spec
- `use_wy=False` is NO LONGER the O(N²) replay kernel. Commit **88212830 "fr13 rewrite sequential tree scan checkpoints"** (HEAD) deleted the two replay loops (~116 lines) and implemented a **single forward walk** with a register-resident `h_cache` tile (L277), parent-resume via `tl.where` select (L280-283), one native delta-rule step per node (L335-339), one store per node (L341-350). The +35.8% O(N²) HBM tax is **structurally gone.**
- **MY SPEC WAS WRONG on beta-bf16.** Native SEQUENTIAL `fused_sigmoid_gating.py:150` `b_beta = sigmoid(b)` is **pure fp32, no bf16.** The "beta bf16 cast" is a WY-path-only artifact. **DO NOT add FLA_BF16_BOUNDARIES/bf16 to `_tree_gdn_kernel`** — it injects drift. (FR13_SEQ_TREE_SCAN_TASK.md step 5 corrected.)

## Op-for-op vs native fused_sigmoid_gating (L136-184): E1-E12 ALL MATCH (zero kernel edits)
gate softplus (β=1, thr=20) MATCH; b_g=-exp(A_log)*softplus MATCH; h0 fp32 seed MATCH; l2norm q,k +1e-6 MATCH; b_q*=scale placement MATCH; decay/delta/β-apply/rank-1/readout all MATCH (fp32 accumulate, in-place, cast-on-store). Parent-resume (L280-283) = native's register-carried b_h on the spine (deepest strict ancestor = parent). BLOCK_V=1 + per-V-row grid makes `b_v*b_k` = native's `b_v[:,None]*b_k[None,:]` sliced.

## THE ONE REAL ACTION (not a kernel edit): E6 — verify the scale wiring
Native spec `scale=None -> head_k_dim**-0.5`. Confirm the caller (`launch_tree_gdn_prepared` L724 / the vLLM patch) passes `output_scale = head_k_dim**-0.5`, NOT 1.0. **If it passes 1.0, q is mis-scaled by √head_k_dim → layer-0 large-nonzero.** This is the single most likely silent-wrong wiring bug — CHECK BEFORE blaming numerics.

## TWO HBM-LEAK items to close for the TPS de-risk (G1/G2)
- **G1 (highest value): accepted-path-only state store.** L346-350 stores `state_i` for EVERY node. Off-path branch states are needed for output logits (produced in registers) but must NOT hit the SSM state bank. Gate the L346 store on the accepted path (reuse `h0_use_accepted_column`/`h0_num_accepted_tokens` at L261-266) — store only when node i is on the accepted path.
- **G2: confirm no h_cache spill.** `h_cache` = [N_PAD≤16, DIM_K=128] fp32 ≤8 KiB/program (BLOCK_V=1). Should be register/SRAM-resident; verify `n_spills==0` via triton.compile/ttgir. Keep BLOCK_V=1; keep the resume loop `tl.static_range` + `tl.where` (never `h_cache[p]` dynamic index, never `tl.load(state/h0)` in the walk except the single b_h0 seed); keep `num_stages=1` on the data-dependent walk (native's num_stages=3 is for the independent-load T-loop — pipelining here re-creates traffic).

## Predicted FIRST live ladder
Layer-0 GDN scan spine → **0.0 by construction** (E1-E12 bit-exact + spine resume = b_h[i-1]; conv bf16-taps + scan static_range fixes already landed upstream). The stale **0.015625 was the WY arm** (f288cba6), NOT a use_wy=False measurement — expect a fresh use_wy=False ladder to read 0.0. WATCH: E6 scale mis-wire (large-nonzero, not a tiny seam — check first); branches via per-depth argmax vs path-rerun oracle (the `tl.where` select is exact → expect 0.0); downstream layer-3 full_attention (0.00195) is a DIFFERENT subsystem — don't conflate with the GDN-scan gate. No self-declare; the live 0.0 ladder + Gate-2 + B=4 CUDA-capture + e2e-vs-E5 + TPS are the gates.
