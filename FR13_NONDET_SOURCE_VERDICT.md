# FR13 — B=4 EXTRA non-determinism: SOURCE bisection (workflow ws5783inp, adversarial-verify holds=True)

3 independent source audits (GDN-scan / FA2-fork / scheduler-RNG) + synthesis + adversarial verify. **Verdict: diffuse-expensive-stop — with 2 cheap residual probes the verify says must run before STOP is final.**

## What the carrier is NOT (source-verified, 3 audits converge)
The two named kernels are **deterministic by construction** and CANNOT produce the 189/256 (74%) divergence:
- **GDN tree-scan** (`fr10_gdn_tree_kernel.py`): only atomic is the COUNT_INVOCATION-gated `invocation_counter` (:250-256, OFF in serving); scan is `static_range` `tl.sum`/`tl.where` over constexpr axes (:277-343); deployed grid `(num_vh, cdiv(dim_v,BV))` (:547) has **NO batch axis** (one launch/seq, serial python loop, disjoint buffers); `num_warps=8` fixed (:582).
- **FA2 fork decode tree-bias** (`fr13_patch_fa2_tree_bias.py:570`): passes paged `block_table` → single-split splitkv; `seqlen_q=tree_len>1` fails the `seqlenq_ngroups_swapped` gate → the num_splits heuristic never runs → `num_splits` stays 0/1 → **per-row reduction order is batch-composition-INDEPENDENT**. The B=1 2-ULP MMA floor (max 0.0039, ~2 ULP/1M, no depth growth) does NOT scale with co-residency.

## What the carrier IS (diffuse, the conv-priorwindow-root front)
**Accept-pattern-dependent state / bank-row selection feeding the deterministic kernels, amplified through the spec-decode feedback loop:**
- conv prior-window READ `clamp(num_accepted-1)` + gather/index_select (`fr10_phase4_patch_vllm_tree_gdn.py ~797-818`) — bank-row selected depends on the (temp-0.6 non-deterministic) prior accept pattern; conv1d_out diverged to 18.375 at call2; **failing row WANDERS run-to-run = the contamination signature**.
- GDN h0 bank read keyed on `num_accepted-1` (`fr10_gdn_tree_kernel.py:261-267`).
- This is the **same diffuse determinism-amplification class FR11 already banked as a no-go** — no byte-exact fix consistent with the no-copy/no-reroute policy.

## The gate FAILS on the RIGHT (seed-robust) metrics — confound-free
- bag-TV **0.2335 > 0.1099** seed-robust budget; real-loss **0.4751** (105 losses outside self-noise); accept/event **2.10 vs native 3.619**; depth-collapse run_len 6; **root-reject 36.6%** (201/549, same single candidate as native = **verify contamination, NOT sampler-flip**).
- **NEW confound-free evidence the synth missed (verify found it on disk):** a clean same-config (64/1) **same-seed (1313)** CUDA-captured native-vs-tree comparison already exists = **165/256 (64.5%) divergence** with NO 128→64 / spp4→1 confound → strengthens the FAIL beyond the confounded 189/256.

## The "53% native floor" is an ARTIFACT → real bar is TIGHTER (worse for the tree)
The "53% floor" (137/256) is `native_bi` seed=1313 vs `native_bi_noise` seed=2313 — a **different-seed sampling** comparison (byte-identical prompts, diverges exactly at the first sampled position), NOT B=4 batch self-noise. BI left positional flips ~unchanged (137→139, seed-dominated) but cut bag-TV 0.152→0.086 (the real batch-order channel). Seed-robust floor ≈ **0.086-0.11 bag-TV**. The genuine same-seed same-regime native self-repeat was **never run**.

## The 2 cheap residual probes (verify: required before STOP is final) — ~2 boots
1. **UNTESTED one-line fix** (verify caught the synth hand-waving "no cheap fix" as a *static* inference): **force `num_splits=1` on the decode `flash_attn_varlen_func`** at `fr13_patch_fa2_tree_bias.py:570` — it currently OMITS it, while prefill `:537` sets `num_splits=1 if VLLM_BATCH_INVARIANT`. Likely inert (carrier is the state-handoff, not FA2 reduction) but never empirically run. ~1 boot.
2. **Same-seed (1313 vs 1313) same-regime native self-repeat** to pin the exact floor (the one genuinely-missing measurement). ~1 boot.

## Honest cost-gate position
Engineering verdict = **diffuse-expensive-stop**: the carrier is the diffuse state-handoff feedback loop (FR11's banked no-go class), not a localizable byte-exact seam, and the tree fails the tighter seed-robust bar confound-free. **Residual caveat:** no live GPU repeat-launch was run here (kernel determinism + "FA2 num_splits inert" are source-level arguments), and the one-line num_splits probe + the same-seed floor probe are cheap. Per research-before-no-go policy, those 2 probes are the honest precondition to a final STOP — they don't change the diffuse diagnosis but they convert "strong argument" → "measured."
