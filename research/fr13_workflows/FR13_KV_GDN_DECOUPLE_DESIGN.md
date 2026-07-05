# FR13 — Decouple full-attn KV cache from GDN recurrent-state cache: feasibility

Date: 2026-07-05. READ-ONLY design pass. Source verified against the live-container
extract under `/tmp/lumo_tree_patch_probe_t11i0n5r/vllm/` (matches the running image the
patcher targets). No GPU/docker touched.

## ADVERSARIAL-VERIFICATION VERDICT (2026-07-05, second pass)

**VERDICT: the literal idea is NOT-FEASIBLE — confirmed against source. The doc's core
reasoning stands; three small corrections below.** Every load-bearing source citation was
re-read this pass and matches: config.py:436-480 (coupling), qwen3_next.py:717-721 (raise on
`'all'`), utils.py:853-891 (`none`=unchanged / `align`=rotating gather at `(seq_len-1)//bs`),
kv_cache_coordinator.py:453-544 (MIN fixed-point + full-attn truncation at :534-540),
single_type_kv_cache_manager.py:778-824 (MambaManager hit empty unless a state block is
cached), kv_cache_manager.py:176-216 (ONE `num_new_computed_tokens` per request),
gdn_attn.py:162-167/199/266-271/305, qwen3_next.py:512-535 (ONE shared residual stream through
all interleaved `linear_attention`/`full_attention` layers), scheduler.py:250-253,
config.py:457-460 (align ⇒ chunked prefill), launcher self-doc (~L112).

**The single fundamental (un-patchable) blocker = ONE `num_computed_tokens` per request**
(kv_cache_manager.py:176-216 → the whole 64-layer stack processes exactly one token slice per
forward). The config policy (config.py:436-480) is only a *patchable* rewrite; the coordinator
MIN is *patchable*; but even after patching both, the single scalar forces the choice
attention-hit-⇒-GDN-**stale** (wrong output) XOR GDN-full-recompute-⇒-**no-TTFT**. You cannot
feed attention the hit-suffix and GDN the full prefix in one pass. That is the wall.

**New confirming fact (strengthens the verdict): `align` restores GDN state by a CHEAP COPY,
not recompute.** gpu_model_runner.py:1447/4030 route align through
`mamba_utils.preprocess_mamba`/`postprocess_mamba` = gather/scatter of the fp32 checkpoint via
`get_mamba_state_copy_func()`. So align already gives correct GDN state at ~zero cost on a hit;
"uncache GDN to force the contiguous recompute path" would be **strictly worse** (pays recompute
it doesn't need) for a state that align already restores correctly. The route flip is a
decode-time **layout/realization** effect, not a state-cost or state-value effect.

**Three corrections applied below** (see ⟪FIX⟫ tags): (1) the pre-GPU bit-exact gate must NOT
be used to *skip* the SWE/route arm on a step-8 fork — §29 shows the route survives autotune-
floor byte differences, so a non-bit-exact Arm C can still route correctly (bit-exact-past-8
CONFIRMS a fix; a fork only fails to confirm — it does not refute). (2) The reframe-patch slot
sketch ("return the leading `1+num_spec` cols") is imprecise: `align` holds the LIVE decode
state in the block gathered at `(seq_len-1)//block_size` (the LAST block), so a `none`-style
override must pin the tree node bank to the row(s) holding the live state, not literally
columns `0..num_spec` of the full-width block table — otherwise it points at the *first* prefix
block and reads a stale/empty slot. (3) The TTFT FLOP percentages (~0.6%) are unverified
model-specific estimates; the *architecturally verified* claim is only qualitative — cached KV
saves the K,V **projection** GEMMs but NOT the attention **output** at prefix positions, which
the shared residual forces you to recompute, so GDN-recompute ≈ full re-prefill regardless of
the exact %.

## TL;DR verdict

The user's idea — "cache full-attn KV (keep TTFT) but let the 48 GDN layers use the
contiguous/recompute decode path (no block-pool) so decode realization matches no-cache and
the round-1 route stops flipping" — is a **DEAD-END as literally specified**, for two
independent reasons, and it also **misdiagnoses the carrier**:

1. **Not expressible in stock vLLM, and not as a config combo.** `enable_prefix_caching`
   and GDN-state caching are two ends of one switch (config.py:436-480 verifier), Qwen3-Next
   forbids the only mode that could abstain (qwen3_next.py:717 raises on `'all'`), the hybrid
   coordinator collapses the combined prefix hit to the MIN across groups so an un-cached GDN
   group drives the attention hit to 0 (kv_cache_coordinator.py:453-544), and one scalar
   `num_computed_tokens` is shared by every group (kv_cache_manager.py:176-216 →
   gpu_model_runner). "KV cached but GDN uncached" is a **non-reachable engine state**.

2. **Even as a patcher add-on it does not keep the TTFT win.** Because all 64 layers share one
   residual stream (qwen3_next.py layer loop), rebuilding a GDN layer's recurrent state over
   the prefix requires the OUTPUT of every lower layer at every prefix position — including the
   full-attn layers' O(L^2) attention. Cached KV lets you skip only the K,V *projection* GEMMs
   (~0.6% of prefill FLOPs); the attention itself must be recomputed. So a GDN "recompute
   shadow" over the prefix ≈ a full re-prefill = re-paying TTFT, not preserving it.

3. **The premise is wrong about the carrier.** The route-probe already ran (campaign §22, N=16
   paired seeds): **native+cache is 16/16 healthy** — i.e. align-mode GDN caching *by itself*
   does NOT flip the route. Only **tree(cat8) × cache** collapses (4/16 delegate, 6/16
   read_file, 6/16 NO_TOOL). So "GDN caching is the route-flip culprit" is a confound: the
   carrier is the **interaction of the tree drafter's node-bank state indexing with align's
   block-pool row layout on turn-1 cold decode**, not GDN caching per se. Decoupling (removing)
   the GDN cache would therefore sacrifice TTFT to kill a cache that is benign in isolation.

**The productive re-frame** (which the campaign is already on, §18/§22/§29): keep BOTH caches
in `align`, and make the **align-mode GDN decode/restore realization route-distribution-
identical to the contiguous (`none`) path** — a lossless-layout problem at the GDN decode-slot
seam, at zero TTFT cost. That is the only version of "decouple the layout from the cache" that
is both buildable and keeps the TTFT win.

---

## 1. Concrete approach — is it a flag, a config combo, or a patcher change?

**Not a flag. Not a config combo. Only a patcher change — and the one that keeps TTFT is NOT
"uncache GDN"; it is "make align-mode GDN decode layout tree-aware and realization-identical
to `none`".**

- `--mamba-cache-mode` exists (config/cache.py:109, arg_utils.py:1025, Literal
  `['all','align','none']`, default `none`) and *looks* like the decouple knob.
- But `MambaModelConfig.verify_and_update_config` (config.py:436-480) **unconditionally
  rewrites** it: if `enable_prefix_caching` is True and mode is `none`, it becomes
  `all`-or-`align` (config.py:437-440); Qwen3-Next lacks `SupportsMambaPrefixCaching` so it
  gets `align`; and `'all'` hard-raises (qwen3_next.py:717-721). If `enable_prefix_caching`
  is False, mode is forced back to `none` (config.py:474-478). The two are mutually implied.
- So `enable_prefix_caching=True + mamba_cache_mode=none` is **unreachable by config**.

## 2. Exact mechanism — three enforcing layers (each verified)

**(A) CONFIG POLICY — the coupling site.** `config.py:436-480`. Verified verbatim:
`if cache_config.enable_prefix_caching: if mamba_cache_mode=='none': mamba_cache_mode = 'all'
if supports_mamba_prefix_caching else 'align'` (:437-440); `else: mamba_cache_mode='none'`
(:474-478). This is the SINGLE place a patch could force the illegal combo.

**(B) COORDINATOR — the MIN.** `kv_cache_coordinator.py:453-544` (verified). Fixed-point:
"each attention type either accepts the current candidate length or reduces it" (:461-464);
simple hybrid = `[FullAttentionSpec, mamba]`, one iteration (:493-495, :531); after
convergence the full-attn blocks are **truncated to the combined hit_length** via
`del blks[num_blocks:]` (:534-540). A `none`-mode mamba group caches no state blocks, so
`MambaManager.find_longest_cache_hit` returns empty (single_type_kv_cache_manager.py:778-824,
verified: it only appends when `block_pool.get_cached_block` hits) → combined hit → 0 →
**attention KV also truncated to 0 → TTFT win destroyed**, exactly the outcome the user wants
to avoid.

**(C) SINGLE SCALAR — one token range per request.** `get_computed_blocks`
(kv_cache_manager.py:176-216, verified) returns ONE `num_new_computed_tokens`; the model
runner stores it as the request's single `num_computed_tokens` and the forward feeds the SAME
suffix token-slice to attention AND to all 48 GDN layers. There is no stock way to give
attention the hit-suffix while giving GDN the full prefix in one forward pass.

**Decode-slot seam (the real, buildable target).** `mamba_get_block_table_tensor`
(utils.py:853-891, verified): `none` returns `block_table` unchanged (fixed leading
`1+num_spec` slots); `align` GATHERS the last `1+num_spec` blocks starting at
`(seq_len-1)//block_size` — a **rotating block-pool row** that migrates as decode crosses block
boundaries. The tree's node bank binds into those rows: `spec_state_indices_tensor =
block_table_tensor[spec_sequence_masks, :num_spec+1]` (gdn_attn.py:266-268), while native
decode uses `block_table_tensor[:, 0]` (gdn_attn.py:199/269). Launcher self-documents it
(fr10_launch_speed_server.sh:114): *"align cache is linear-layout, native-aware but NOT
tree/node-bank-aware."* This is where align's layout collides with the tree's state indexing.

## 3. Why "decouple GDN" would (not) remove the round-1 route flip

- The user's mechanism ("GDN decode realization now matches no-cache") WOULD, if realizable,
  make GDN decode identical to the no-cache arm that resolves. But:
- **native+cache is 16/16 healthy** (§22) with GDN cached in align → GDN caching alone doesn't
  cause the flip. Removing the GDN cache is aimed at the wrong object.
- The flip is **tree × align** (§18/§22): all six tree+cache variants collapse (config-
  independent → systematic logit shift), native+cache and cat8-nocache stay 16/16.
- §29: the collapse persists in **EAGER** (3/16) as well as GRAPH (4/16) → it is **real
  numerics**, not solely the captured-graph baked indexing.
- §25/§35: turn-1 cache-vs-nocache is **bit-exact through decode step 7** (torch.equal,
  max_abs 0.0 across all 48 GDN layers); the route diverges DISTRIBUTIONALLY at step 8+.
- §29 control: nocache-resend (autotune floor) already differs byte-wise 16/16 **yet stays
  16/16 healthy** → the route is robust to the floor; the cache-config produces a *systematic*
  step-8+ shift the floor does not. So the carrier is the align **storage layout** (block-pool
  strides / fp32 checkpoint dtype / rotating row) driving a systematic post-step-7 fork under
  the tree decoder, not the state VALUES (identical through 7) and not autotune noise.

⇒ A patch that forces the tree's GDN decode to a **contiguous, `none`-style fixed node-bank
slot** (while attention keeps paging and the align checkpoints still exist for cross-turn
restore) removes exactly that layout difference on turn-1, where there is no hit to preserve.
It should return the route to `agent` **iff** the layout is the carrier. Confidence: **medium**
— bit-exact-through-7 means the state values are already identical, so causality runs through
kernel-dispatch/accumulation layout, which is plausible (cf. L0c autotune-drift) but unproven
until Arm C below is run.

## 4. TTFT tradeoff

- **"Uncache GDN + keep KV" (the literal idea): TTFT win = ~0.** Shared residual forces
  full-attn O(L^2) recompute over the prefix; cached KV saves only the K,V projection
  (~0.66% at 15k / ~0.62% at 32k / ~0.60% at 40k of prefill). GDN's own recurrence roll is
  ~1% of the GDN mixer and fixed-size, so GDN IS cheap in isolation — but it is not isolable.
  Net ≈ full re-prefill, and prefill is **compute-bound** (760 TFLOP@15k → 2225@40k), i.e. the
  real wall-time TTFT that prefix caching exists to remove. The "free compute on loaded
  weights" steelman conflates HBM-bound decode with compute-bound prefill; at B>1 the recompute
  also contends for the same weight-read bandwidth as live decode.
- **The productive re-frame (make align GDN decode layout = `none`): TTFT tradeoff = ZERO.**
  Both caches stay in `align`; cross-turn KV reuse and GDN checkpoint restore are untouched;
  the change is only the decode-time *slot indexing* of the tree node bank. Turn-1 has no hit,
  so nothing is forfeited. This is why the re-frame, not the decouple, is the right lever.

## 5. Risks / open questions (answering the two the user asked)

- **Does vLLM allow KV cache without mamba-state cache for a hybrid? NO.** Structurally
  impossible: verifier (A) + coordinator MIN (B) + single `num_computed_tokens` (C). Forcing it
  by patch collapses the attention hit to 0 (no TTFT) and, on any spuriously non-zero hit,
  reads stale GDN state (`has_initial_state = context_lens>0`, gdn_attn.py:305;
  `initial_state = ssm_state[non_spec_state_indices_tensor]`, gdn_linear_attn.py:830) → wrong
  output.
- **Does GDN recompute over a cached-KV prefix produce a consistent state? Only if you
  re-forward the full stack.** A GDN-only recompute that reads cached KV still needs each
  full-attn layer's OUTPUT at prefix positions (shared residual) → must recompute attention;
  skip it and the GDN state is inconsistent. Caching the per-layer residual instead of KV would
  fix consistency but costs more than the KV cache (>5 GB/req at 32k) and is not what vLLM does.
- **Decode-slot patch risks (the productive path):** (i) causality not guaranteed
  (bit-exact-through-7); (ii) `align` requires `enable_chunked_prefill` (config.py:457-460) and
  a block-aligned scheduler split (scheduler.py:253 `need_mamba_block_aligned_split`) — a
  slot override must stay consistent with both; (iii) the FULL decode CUDA-graph bakes
  capture-time GDN indexing (launcher:125-127) — a decode-slot override must match graph
  capture or it writes restored state to the wrong row (note §29 shows the flip persists in
  eager, so graph indexing is not the sole carrier, but the patch must not *introduce* a
  graph/eager mismatch); (iv) spec-decode: the override must preserve the tree node-bank rows
  (`spec_state_indices_tensor[:, :num_spec+1]`) and not clobber cross-turn restore, which reads
  the align checkpoint rows the decode path must have written.

## 6. DECISIVE EXPERIMENT — minimal, reuses the existing route probe

The harness already exists: campaign §22/§29 ran a 3-arm / 2×2 paired-seed turn-1 route probe
(reduces: `research/fr13_workflows/fr13_route_probe_3arm_reduce.txt`,
`fr13_route_probe_2x2_reduce.txt`). Add ONE arm.

**Route-probe, N=16 paired seeds, temp 0.6, B=1, SAME astropy-13453 turn-1 request, cat8 tree:**
- Arm A (control): cat8 **no-cache** — expect 16/16 delegation (agent). [known]
- Arm B (repro):    cat8 **+cache** (align, conv-baked default) — expect ~4/16. [known]
- Arm C (test):     cat8 **+cache + GDN-decode-contiguous-slot patch** — force the tree node
  bank to a fixed `none`-style slot at decode. ⟪FIX⟫ **Precise seam:** in `align`, the LIVE
  decode state for the tree lives in the block that `mamba_get_block_table_tensor`
  gathers at `(seq_len-1)//block_size` (utils.py:879-891, the LAST block), which
  gdn_attn.py:266-268 then narrows to `[:, :num_spec+1]`. The override must make that gather
  return a **fixed contiguous row that both the write and the next restore agree on** — i.e.
  pin the tree node bank to the same physical slot every decode step (contiguous, `none`-like),
  NOT literally `block_table[:, 0]` of the full-width align table (that column is the FIRST
  prefix block and holds stale/empty state). Concretely: override the align branch of
  `mamba_get_block_table_tensor` for spec/tree requests to reuse the previous step's gathered
  slot indices (or a dedicated fixed node-bank slot), leaving the align **checkpoint** gather
  used by cross-turn restore untouched. **Metric = turn-1 tool route distribution** (delegation
  vs read_file vs NO_TOOL) over the 16 paired seeds.
- Decision: **Arm C ≈ Arm A (≥14/16 delegation) ⇒ the layout is the carrier and the productive
  re-frame works at zero TTFT cost.** Arm C ≈ Arm B ⇒ layout is not the carrier; the flip is a
  deeper tree×align-prefill distributional effect and the literal "uncache GDN" idea is doubly
  dead (impossible AND wrong target).

**Cheapest pre-GPU gate** (extends the §25/§35 in-process same-boot bit-exact probe, no SWE
arm): under one boot, capture GDN decode outputs steps 0-15 for Arm C vs Arm A with
`torch.equal`. If Arm C stays bit-exact PAST step 8 (where Arm B forks), that CONFIRMS the route
fix before spending a full 16-seed SWE arm — proceed to ship the slot patch. ⟪FIX⟫ **But a
step-8 fork does NOT refute the patch and must NOT be used to skip the route arm.** The gate is
sufficient, not necessary: §29 already showed the route is robust to autotune-floor byte
differences (nocache-resend differs 16/16 byte-wise yet stays 16/16 healthy), so Arm C can be
NON-bit-exact to Arm A past step 8 and still route to `agent`. Bit-exact-past-8 ⇒ strong
positive (ship); fork-at-8 ⇒ **inconclusive** ⇒ still run the 16-seed route arm. Only skip the
route arm if Arm C reproduces Arm B's *token-level route* on the in-process probe (same argmax
tool token), not merely a byte fork.

## Cited source (all verified this session, live-container extract)

- config.py:436-480 — coupling policy (enable_prefix_caching ⇄ mamba_cache_mode). VERIFIED.
- qwen3_next.py:717-721 — NotImplementedError on `'all'` → forced `align`. VERIFIED.
- kv_cache_coordinator.py:453-544 — MIN fixed-point; full-attn truncated to combined hit
  (:534-540). VERIFIED.
- single_type_kv_cache_manager.py:778-824 — MambaManager hit empty unless state blocks cached.
  VERIFIED.
- kv_cache_manager.py:176-216 — single `num_new_computed_tokens`. VERIFIED.
- utils.py:853-891 — `mamba_get_block_table_tensor`: `none`=unchanged fixed slots,
  `align`=rotating gather at `(seq_len-1)//block_size`. VERIFIED.
- gdn_attn.py:199, 266-271, 305 — non_spec (`[:,0]`) vs spec node-bank (`[:, :num_spec+1]`);
  `has_initial_state = context_lens>0`. VERIFIED.
- fr10_launch_speed_server.sh:114-127 — "align native-aware but NOT tree/node-bank-aware";
  captured-graph persistent indexing. VERIFIED.
- Campaign §18/§22/§29/§25/§35 — native+cache 16/16 healthy; only tree×cache collapses; eager
  reproduces; turn-1 bit-exact through step 7; nocache-floor route-robust. VERIFIED (doc).
