# Why option 1 (snapshot-side fix) failed — CORRECTED (2026-06-20)

Branch `fr13-prefix-cache`. **This supersedes the earlier "cohort-coverage" thesis, which was WRONG.**
The data (mined from the `unidiag2` drill log + the apply-mechanism workflow wd8yxwms3) tells a
different and clearer story. Keeping the corrected record so we don't relapse.

## What the data actually shows
Mining `output/fr13_bigdenom_swe/cat9_apc_unidiag2/docker_full.log` (FR13_APC_SSM_SNAPSHOT=1, the
substitution ACTIVE):
- The **postprocess snapshot copies — the actual carrier — DO find the committed leaf** every time:
  `preproc=False leaf=80 will_sub=True stock_src=87` (and 174 vs 199, 282 vs 316). Not a miss.
- The 48/50 `leaf=None` I had agonized over are all **preprocess** copies (`bias>0`). Preprocess is
  *restore* (reads FROM the cache), so a None there is harmless — it's not the snapshot carrier.
- **Yet the agent still gave up empty** (`empty patch (agent_gave_up)`, verdict=failed) with the
  substitution computing `will_sub=True`. So the leaf was found and the substitution was *computed*,
  but the snapshot was **still poisoned**.

## The real reason option 1 failed
The substituted `MambaCopySpec.start_addr` (pointer redirected to the committed-leaf row) **had no
effect on the copied value.** The apply step copied the stock row anyway. Per workflow wd8yxwms3
(2 independent readers of the live vLLM source), the apply path is:
`collect_mamba_copy_meta` stores `copy_spec.start_addr` into a CPU pointer buffer →
`do_mamba_copy_block` → Triton `batch_memcpy_kernel` does `tl.load(src_ptr)` at apply time. The apply
**honors start_addr** and reads the **value at apply time**, synchronously after collect. So in
principle the SUB *should* have worked — which means it silently **did not execute / did not reach
the buffer** (a patch-level failure), OR there is a second carrier. Either way the snapshot-side
pointer-redirect is **not a reliable lever**: it depends on the reassigned spec actually landing in
the pointer buffer, and in the live run it didn't.

So option 1 did NOT fail on "the align cohort is disjoint from the committed set" (it isn't — the
carrier finds the leaf). It failed because **redirecting the copy_spec pointer was computed but
ineffective.**

## Why option 3 (write-through) is the robust version
Same fix, lower layer. Instead of redirecting the read pointer, **overwrite the VALUE** at the exact
block-pool row the stock snapshot reads:
`state[block_ids[src_block_idx + accept_token_bias]].copy_(state[committed_leaf_row])` at collect
time. Because the apply reads the value at apply time and runs synchronously after collect (same CUDA
stream, no fence — wd8yxwms3, both readers agree), the snapshot then copies the corrected value —
**independent of whether any substituted copy_spec pointer takes effect.** It also covers every
committed req at the source, so the (irrelevant) align cohort can't matter. Implemented gated
`FR13_APC_SSM_WRITE_THROUGH` (default 0 → byte-identical); the old pointer-SUB is disabled when the
write-through is on, so a drill cleanly attributes the result. Gate = **black-box** (agent full wall +
no garble), NOT Tap-C: Tap-C is **index-based** (`is src_row in committer's written rows`) and the
write-through deliberately keeps the stock index while fixing its value, so Tap-C will always read
"stale." That index-vs-value confusion likely misdirected the whole option-1 effort.

## Scope: who actually has this bug (important for the baseline)
The stock align formula `state[block_ids[cur_block_idx + num_accepted_tokens - 1]]`
(`vllm/model_executor/layers/mamba/mamba_utils.py:285`) was **designed for native MTP's linear
layout**: native spec-decode writes each speculative position's recurrent state to *consecutive
block-pool slots*, so the accepted leaf genuinely sits at `block_ids[cur+num_accepted-1]`.
- **Non-spec decode**: NO bug — `num_accepted=1` → `block_ids[cur]` = where the state lives.
- **Native MTP spine**: NO SSM-snapshot bug — the align formula matches native's block layout.
- **Our tree committer**: HAS the bug — we write the accepted leaf to a **NODE BANK**
  (`spec_state_indices`) because a *tree has branches* and the linear "position k → block slot k"
  layout doesn't hold. So the align reads a stale block slot.

Therefore the write-through is not a hack around a self-inflicted wound — it **re-establishes the
native invariant** (leaf back in `block_ids[cur+num_accepted-1]`), making our tree+APC behave exactly
as native MTP+APC already does. (Separate, out-of-scope here: the spec-intrinsic conv-window carrier
SGLang #25587 / vLLM #43559, present with or without APC.)

## One-line takeaways
- The carrier (postprocess snapshot) **finds the leaf**; option 1 failed because the pointer-SUB was
  **computed but ineffective**, not because of cohort coverage (earlier thesis retracted).
- The apply honors `start_addr` and reads the value at apply time, synchronously after collect →
  a collect-time **value write-through lands** (wd8yxwms3, 2 readers).
- Gate the write-through on the **black-box**, not Tap-C (index-vs-value).
- The bug is **our-tree-specific** (node bank for branches); non-spec and native MTP are unaffected;
  the write-through re-establishes the native block-layout invariant.

Cross-refs: [[apc_ssm_carrier_deep_findings]], [[apc_ssm_drilldown_design]],
[[sglang_mamba_radix_cache_design]], [[project_fr13_conv_priorwindow_root]].
