# FR-13 — e2e miss root-caused to the prefill bug; grind drift to 0 (spine + branch), then re-e2e

User directive 2026-06-08: NO copy / reroute / splice / copy-recurrent pivot (banned reward-hack). The deficit is our kernel's drift — drive it to **0** in our own kernel, spine AND branch, then chase the e2e.

## The e2e miss (the deliverable result that triggered this)
Run `output/fr13_argmax_e2e_20260608T055851Z`, forked-FA2 B=4, tree ENGAGED (286/288 steps `has_tree_parent_indices=True`; 924 committer events; non-vacuous):
- **accept/event = 1.11 vs E5 3.076 — a 2.7× MISS.**
- accepted_len histogram: **0→534 (57.8%)**, 1→104, 2→125, 3→48, 4→37, 5→76. Bimodal: when the draft matches the tree's verify argmax it fully accepts (len 5); 58% of events the tree's verify argmax rejects the draft's first token.
- Same ballpark as the prior TREE_ATTN-Triton build (2.8× miss). ⟹ **full-attn byte-exactness (the CUTLASS fork) was necessary but NOT sufficient; the remaining lossy source is the GDN spine drift.**

**PAIRED native arm (same harness, `native_mtp5/quick_native_mtp5_b4.json`):** native MTP-5 accept/event = **3.21** (1597/497), warm TPS 15.6, per-req TPS 47.3. So tree (1.11 / 2.67 / 8.78) loses **2.9× on accepts AND ~5× on speed**. Native = 3.21 ≈ E5 3.076 ⟹ harness sane. Same drafter family + same prompts + native accepts 2.9× more ⟹ the **tree VERIFY is lossy** (not the drafter) — confirms the drift→acceptance link.

## Root cause = the prefill bug (first nonzero in the spine ladder)
Spine ladder (`FR13_LADDER_LOG.md`, run `fr13_ex2_live_ladder_20260608T021853Z`): first nonzero is **L8 `linear_attention` 0.0039**; its only nonzero input is **`h0_state_in` = 7.2e-4** (conv/pre_conv/conv1d_out all 0.0). That `h0_state_in` is the GDN recurrent-state seed written during **prefill**, inherited from the forked-FA2 **L7 prefill** divergence (`FR13_PREFILL_DRIFT.md`): tree prefill calls `unified_attention` with `q_descale=None` + `descale_shape=(...,key.shape[1])` instead of native `flash_attn_varlen_func` with `q_descale=layer._q_scale.expand(...)` + `(...,num_kv_heads)` + `scheduler_metadata`/`fa_version`/`num_splits`. Compounds L8 0.0039 → 1.9 final logits → argmax flips at low-gap positions → 58% step-0 rejects.

## The fix (written, ready)
`patches/fr13_fa2_prefill_native.patch` — flag-gated `FR13_FA2_PREFILL_NATIVE` (default OFF), routes TREE_ATTN prefill through `flash_attn_varlen_func` matching native call-for-call. `git apply --check` clean.

## Grind plan — chase to 0, spine AND branch (no asking per front; report at the e2e win or a true wall)
1. **Finish the in-flight native arm** (codex, one-GPU) — confirms 3.076 on this harness + localizes (at the tree's step-0 rejects, does native accept the same draft? = verify-lossy confirmation). Then tear down + `recover_host_memory`.
2. **Apply** `patches/fr13_fa2_prefill_native.patch` to the live `scripts/fr13_patch_fa2_tree_bias.py`; commit. Relaunch forked-FA2 server with **`FR13_FA2_PREFILL_NATIVE=1`** (verify the patcher anchor matches the container's installed `tree_attn.py` — fail loud if not).
3. **Re-run the top-down spine ladder** vs native (eager B=1, fallback UNSET): confirm **L7 `attn_out_raw`=0.0, L8 `h0_state_in`=0.0, and the WHOLE ladder input→every layer→final logits = 0.0.** If a fresh nonzero appears at a deeper GDN layer (previously masked by contaminated inputs), localize the sub-op (conv/scan/gate/o_proj/state-handoff) and drive it to 0 with our kernel per the bit-exact methodology (conv=ex2.approx, scan=tl.range static-range, etc. — NOT copy/splice). Repeat until the spine ladder is 0.
4. **Re-run the branch oracle ladder** (native-FA2-on-branch-path; SpecInfer Def 4.1 / STree Eq.4-6): confirm branches 0.0 (within the FA2 floor; the L55 4.9e-4 is within floor). Grind any branch-specific fresh source to 0.
5. **Re-confirm Gate-2** (forked FA2 no-bias regular decode == pristine = 0.0) — the prefill flag must not touch regular decode. Bind both gates + per-layer max_abs to `FR13_LADDER_LOG.md` per commit.
6. **Re-run the e2e** (B=4, CUDA-graph, same8): once spine+branch drift = 0, superset-by-math gives accept/event ≥ E5 3.076 + bag-TV ≤ floor. Bring the numbers to the user (do not self-declare pass/fail).

## Banned (user, standing): copy / state-copy / reroute / splice / dense / copy-recurrent multi-spine. Our kernel computes; verify vs native-on-path oracle (splice OFF).
