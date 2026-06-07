# FR-13 (codex_fr13) — drive the full_attention per-layer drift to 0 (TREE_ATTN vs FLASH_ATTN)

**Branch:** main · ONE GPU, serial · Continuous-fix (no asking between fixes). Read `FR13_REDTEAM_GAP_LOCALIZATION.md` (the localization, committed) first.

## Where we are (localized, no hand-wave)
The E5 deliverable failed: tree accept/event **0.92 vs E5 2.61**, TPS **4.8 vs 16.5**, and the tree is **LOSSY** (16/64 samples emit spurious EOS the target ~never does; bag_TV(tree,E5)=**0.558** vs E5 self-floor **0.059**; first_token_TV=0.0 → divergence accumulates with depth).

Cause is pinned to ONE subkernel. From `output/fr12_propagate_compare_20260606T203056Z/layer_hidden_spine_direct_compare.json` (eager spine, tree vs native MTP-5):
- input + layers 0,1,2 (GDN) = **0.0** (bit-exact).
- **first nonzero = layer 3 = first `full_attention`, max_abs 0.0040.**
- EVERY full_attention layer `[3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63]` injects drift; it compounds **0.004(L3) → 0.078(L19) → 0.34(L27) → 0.75(L59) → 7.25(L63)**; final_norm 0.75.
- The deep *linear_attention* layers (60,61,62 = 0.15/0.18/0.58) drift ONLY because their residual-stream input is already contaminated — GDN kernels are bit-exact, so they are **symptoms, not sources.**

**Conclusion: the sole source is the `full_attention` subkernel — TREE_ATTN (our tree path, exp2) vs FLASH_ATTN (native).** The WIRING (depth-RoPE q/k_after_rope→0, ancestry mask) was already fixed in prior passes; the residual ~0.004/layer is an op-level numeric difference that compounds over 16 layers → lossy.

## Goal: full_attention per-layer drift = 0 on spine AND branch, all 64 layers + final logits
"Make it right" = drive the per-layer drift graph above to 0 everywhere (currently 3/64 layers are 0). When full_attn = 0, the GDN-symptom drift and the final-logit drift vanish → lossless.

## Method (continuous; ONE GPU; eager diagnostic; splice-OFF; spine AND branch)
1. **Reconfirm the WIRING is clean first** (so you don't build a kernel for a wiring bug): capture layer-3 full_attn `q_after_rope`, `k_after_rope`, positions, and the attention mask for tree vs native on the SAME decode event. q/k_after_rope should be 0.0 and the mask must be the correct ancestry mask. If either diverges → it's WIRING (position/mask/layout); fix the wiring → recheck.
2. **If wiring is clean, localize the KERNEL op.** Capture, in order, for layer-3 full_attn: `qk_scores` (pre-softmax), `softmax/P` (post-softmax probs), `P@V`, `attn_out`. Find the FIRST op where tree (TREE_ATTN `triton_unified_attention`) diverges from native (FLASH_ATTN `_vllm_fa2_C.varlen_fwd`). Read BOTH kernels' live source (`tree_attn.py` / `triton_unified_attention.py` vs `flash_attn.py` / the vllm-flash-attn csrc). Candidate op-level diffs to check: softmax-scale placement (scale-after-QK vs folded `softmax_scale*log2e`), exp base (exp2 already applied? verify it actually changed the per-layer attn_out — FR13 never re-measured), KV-block iteration order (reverse vs forward), online-softmax rescale, fp32 accumulation vs bf16 cast boundaries on QK and P@V, qk input dtype.
3. **Fix the diverging op in OUR kernel** to match FLASH_ATTN → that op's output = 0.0 (spine+branch). Then re-propagate the full stack and confirm: ALL 64 layers + final_norm + final logits = 0 on spine AND branch (the graph above, but all-zero).
4. **WIRING vs KERNEL discipline (user):** WIRING (mask/pos/layout/backend-config) → fix wiring. KERNEL (native kernel computes the wrong thing for the tree, OR TREE_ATTN/FLASH_ATTN op numerics differ) → align OUR kernel. Do NOT build a kernel for a wiring bug; do NOT reward-hack (splicing native attn is oracle-ONLY, every 0.0 verified splice-OFF).
5. **If — and only if — the op-diff is proven IRREDUCIBLE** (a real algorithmic difference, not an alignable numeric choice): the fallback is FLASH_ATTN + tree-mask (bit-exact full-attn vs E5). Do NOT touch FLASH_ATTN until TREE_ATTN is confirmed dead. Note TREE_ATTN now CUDA-graph-captures (the metadata-builder fix landed), so it is a viable backend if its numerics can be aligned. Report the exact op + your WIRING-vs-KERNEL verdict to the user BEFORE taking the FLASH_ATTN fallback.

## Constraints
- ONE GPU job at a time; kill leftover containers + stuck health-loops; relaunch crashed captures WITHOUT --rm; `torch.cuda.empty_cache()`; boot-free/eager probes where possible.
- Read LIVE vLLM source (`/tmp/vllm-0.22-src`, `/tmp/vllm_live_019`) before patching. Probe weights `/models/qwen3.6-27b-fp8`.
- Commit + push every real step to main; numbers in committed docs (`output/` is gitignored).
- Verify SPINE *and* BRANCHES on every 0.0 (branch oracle = native-on-branch-path; `reference_gdn_tree_branch_oracle_losslessness`).
- Ask ONLY before a close/pass-fail verdict or a copy/dense/re-stream/reward-hack shortcut (those remain BANNED). Otherwise proceed continuously.

## Definition of done
full_attention per-layer drift = 0 (spine+branch) at layers 3..63, → ALL 64 layers + final logits = 0 → re-run the E5 deliverable (B=4 captured SWE-4): lossless (within E5 floor 0.059) + accept/event vs E5. Bring numbers before any closeout.

## RED-TEAM UPDATE (Claude, 2026-06-07) — the residual op is attn_out_raw 0.00195; the 7.25 was stale
Tracing the existing L3 captures (output/fr12_full_attn_l3_*_compare_*/full_attn_l3_spine_compare.json):
- Wiring is DONE: position_ids 2.0->0.0, q/k_after_rope 4.5/1.8->0.0/0.0 (depth-RoPE fix).
- The remaining divergent op is the ATTENTION CORE: `attn_out_raw` = 0.00195 in the treeattn(ancestry-mask) path [rounds to o_proj_out 0.0 AT L3], vs 0.427 in the flashpath (FA2 varlen carries no tree mask) and 0.427 pre-ancestry-mask.
- The 7.25@L63 (fr12_propagate_compare_203056Z) is STALE — it predates the ancestry-mask fix (timestamps 20:30 < 22:05). But the POST-fix propagate (fr12_propagate_treeattn_compare_221323Z) STILL hits ~4.75@L63: the per-layer 0.00195 residual COMPOUNDS ~1.6x/layer over the 16 full-attn layers -> lossy e2e. Per-layer within-floor != e2e within-floor.

**So target `attn_out_raw` (the TREE_ATTN softmax/P@V core), not the stale 7.25.** Steps:
1. Re-run a FRESH full propagate with the CURRENT deliverable config (TREE_ATTN + ancestry mask + exp2) to get the true per-layer drift across ALL 16 full-attn layers (the 203056Z is stale). Check the LATER full-attn layers (7,11,..,63), not just L3 — confirm the compounding.
2. Decide the 0.00195 attn_out_raw: ALIGNABLE (an op choice in tree_attn.py/triton_unified_attention.py vs flash_attn.py: exp base [is exp2 actually applied at L3? re-measure], softmax-scale placement, KV-iteration order, online-rescale, fp32-accum vs bf16-cast boundary on QK and P@V, qk dtype) -> align OUR kernel -> attn_out_raw 0.0 -> propagate 0. OR IRREDUCIBLE (Triton warp-MMA reduction order can't bit-match CUTLASS FA2; memory reference_treeattn_cudagraph_and_fa2_numerics says 0.00195 is the bf16 floor) -> then TREE_ATTN cannot be e2e-lossless because the floor compounds; the lossless route is FA2 + a tree ancestry mask IF FA2's kernel can accept one (the flashpath 0.427 shows the stock FA2 varlen path does NOT) — report this to the user before any FA2 work.
3. If alignable: drive attn_out_raw 0.0 on spine AND branch at L3, then ALL 16 full-attn layers, then confirm the fresh propagate = 0 at all 64 layers + final logits.

**Do NOT conclude "L3 o_proj=0.0 so full-attn is fixed" — that is the hand-wave; the e2e is still lossy (0.92, bag_TV 0.558). The compounding over 16 layers + 64 decode positions is the real target.**
