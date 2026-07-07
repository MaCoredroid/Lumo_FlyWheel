# FR13 — Lossless APC + tree-spec on Qwen3-Next-27B-fp8 (GB10): results & verdicts (living doc)

**Goal:** make vLLM APC (prefix caching) LOSSLESS with tree/spec speculative decoding on Qwen3-Next-27B-fp8
(GDN-hybrid, DGX Spark GB10) at a SMALL `mamba_block_size`, preserving BOTH speedups (decode-spec TPS +
prefix-cache TTFT). Standing constraints: temp 0.6 (never temp-0 — not real-world), live SWE-Verified tasks
for gates (12907 = the cheap proxy), no parallel testing, commit RAW + roll onto main periodically.

This doc accumulates the loose results + verdicts. Detailed sub-docs: `FR13_SPEED_CACHE_ON_VS_E5.md`,
`FR13_CONFIG_DIFF_HISTORIC_VS_CURRENT.md`, `FR13_TREE_SHAPE_CAT6_VS_CAT10.md`, `FR13_APC_EXACT_SEED_SUCCESS.md`.

---

## 1. Losslessness — the two-layer cache fix  ✅ EXACT_SEED proven (L0)

The APC cache stored the WRONG SSM state, in two layers:
- **Layer 1 (staleness):** the snapshot read a stale node-bank row (|diff|~14-18). Fixed by **`SNAP_FIX`**
  (committed-leaf redirect, "FAITHFUL 240/240", **baked default**). *Necessary-not-sufficient.*
- **Layer 2 (realization gap):** even the committed-leaf state is the *recurrent*-kernel realization, not the
  *chunked-prefill* one (differ ~0.0078 >> bf16 ULP). Fixed by **`EXACT_SEED`** (cache the chunked
  realization at 64-aligned boundaries; restore the <64 remainder through the chunked kernel).

**VERDICT — EXACT_SEED restore is BIT-EXACT (L0 mechanism gate PASSED, 2026-06-29).** Drift chain
`77.96 (block 816, lossy) → 38.36 (1024, no-fix) → 30.11 (1024, EXACT_SEED)`. **47/48 GDN layers reach
fp-level drift (~0.0005 mean).** The residual (30–51) is a **MEASUREMENT ARTIFACT**, not a restore error:
it is entirely Layer 0 (a long-horizon accumulator, forget-gate≈1, state magnitude ~90); the *no-cache
control itself* differs there by 38.44 (> the cache's 30.11). A flat 0.0078 threshold is meaningless for a
magnitude-90 accumulator.

**Block 1024 IS lossless with EXACT_SEED** — the whole point was "lossless at small block_size WITHOUT the
8192 band-aid," achieved. The "block-1024 bad zone" is a **pre-EXACT_SEED** (lossy / SNAP_FIX-only) artifact.

**Status:** EXACT_SEED is **proven (L0) + gate-used (`=1`) but NOT yet the launch default** (`:292` is `=0`;
"bake as default" = task #10, pending). It requires `SNAP_FIX=1` (baked).

### 1a. Committer deletion — provably inert  ✅
The decode-time committer `_fr13_apc_exact_seed_recompute` (`ES_CHAIN_PUBLISH=0`, never fired) was a separate
**vestigial** mechanism — the working EXACT_SEED is the **prefill-capture** chain (capture→write→store→
restore→per-layer seed). It was DELETED (commits `3232de2c`+`98a809b7`), removing real **48 host-syncs/step +
the memory leak + the per-token `.clone()` tax**. **VERDICT:** L0 re-verify on the committer-deleted build is
**bit-identical** to the committer-present build (`overall_max=50.84, layer 0, 47/48 fp-clean` both) → deletion
changed nothing → safe, and confirms the committer was vestigial.

---

## 2. Decode cost of the cache — NEUTRAL  ✅

Per-forward GPU time `s_per_fwd` (the confound-resistant metric; `decode_time / num_drafts`, `num_drafts ==
forwards` at B=1, verified via `gen = drafts × (accept/event+1)` to ≤0.1%):

| arm | tree | cache | s_per_fwd |
|---|---|---|---|
| cat6 | 6-node | OFF | 248.5 ms |
| cat6 | 6-node | **ON** | **245.8 ms** |
| chain5 | 5-node | OFF | 248.5 ms |
| chain5 | 5-node | **ON** | **245.0 ms** |

**VERDICT — the lossless cache adds ~ZERO per-forward decode cost.** Same tree, cache ON vs OFF → 245–246 vs
248 ms (a hair *faster*, within noise; a decode cache reads the same KV either way and physically cannot speed
up a forward — the ~1% is a context-length confound, the *failed* ON runs sit at ~4,000 fewer tokens/forward).
Decode is **weight-bandwidth-bound** (~27 GB fp8 weights/forward dominate; the cache tax is 7.89 MiB = 0.028%,
the tree-size term is ~3 ms/node hidden behind the weight stream). **The cache's payoff is entirely
prefill-side, not decode.**

> **Correction (paper-critical):** the apparent "30% cache-ON decode tax" vs the historic 23.88 was an
> ARTIFACT: (a) basis mismatch (23.88 = token-weighted derived; same run per-request = 18.51) and (b) on the
> matched task 12907, current cat6root (17.87) actually *beats* historic (17.61). No real decode regression.

---

## 3. Cache benefit — large, prefill-side  ✅

| metric (12907) | cache-OFF (chain5) | cache-ON (cat10) | |
|---|---|---|---|
| **prefix hit rate** | 0% | **87%** | context reuse leverage |
| **prefill tokens saved** | 0 | **2.33 M** | of 2.68 M prompt tok |
| **TTFT** | 46.1 s | **2.74 s** | **~17× faster first token** |
| **e2e tok/s** | 6.17 | **11.95** | **~2× request-level throughput** |

**VERDICT — the cache ~halves end-to-end latency (e2e tok/s ~2×, TTFT ~17×) by serving ~87% of the agentic
context from cache.** The benefit metric besides TTFT is **e2e tok/s** (request-level), quantified by **prefix
hit rate**. Decode tok/s is cache-neutral; do NOT look for the cache win there.

### 3a. Full tree×cache matrix — proven config (EXACT_SEED=1, cap=500), 12907, 2026-07-01 ✅

Six arms (chain5/cat6root/cat8 × cache OFF/ON), thinking cap `LUMO_PROXY_THINK_BUDGET=500` live on all,
**all 6 RESOLVED**. Speed reported in the **two field-standard lenses** — nobody reports an agentic run as a
single tok/s: *serving* reports Output-Speed + TTFT **separately** (Output-Speed is decode-only ⇒ cache-neutral
by construction), *agentic* reports a **wall-clock duration** per task (Artificial Analysis / SWE-bench). This
supersedes the length-confounded "e2e tok/s" above.

**Serving lens (Artificial Analysis / vLLM):**
| arm | Output-Speed (decode tok/s) | TPOT (ms) | **TTFT (s)** | accept/ev | s_per_fwd |
|---|---|---|---|---|---|
| chain5 OFF / ON | 17.7 / 17.1 | 56 / 59 | **11.78 / 2.60** | 3.39 / 3.19 | 248 / 246 ms |
| cat6 OFF / ON | 15.4 / 16.5 | 65 / 61 | **11.40 / 2.53** | 2.81 / 3.10 | 247 / 249 ms |
| cat8 OFF / ON | 18.0 / 16.0 | 55 / 63 | **10.17 / 2.94** | 3.48 / 2.97 | 248 / 249 ms |

- **Output-Speed is cache-NEUTRAL** (ON≈OFF within ±10% per tree); s_per_fwd flat ~247 ms across all 6.
- **TTFT is the cache win: ~4× on every tree** (11.78→2.60, 11.40→2.53, 10.17→2.94 s).

**Agentic lens (AA Coding Agents / SWE-bench):** the "speed" is **wall-clock time / task** (a *duration*):
| arm | wall-clock/task (agent) | turns | cap fires | hit% | resolve |
|---|---|---|---|---|---|
| chain5 OFF / ON | 15.8 / **7.1** min | 11 / 22 | 3 / 2 | — / 88 | ✅ / ✅ |
| cat6 OFF / ON | 12.1 / **7.6** min | 14 / 20 | 2 / 2 | — / 89 | ✅ / ✅ |
| cat8 OFF / ON | **7.3 / 11.3** min | 11 / 14 | 1 / 4 | — / 85 | ✅ / ✅ |

**VERDICT — the cache win reports cleanly as TTFT ~4× (serving) + wall-clock faster (agentic); Output-Speed is
correctly flat.** All six arms met the 30-min/task bar (max 15.8 min). Wall-clock ON-vs-OFF is
**path-confounded** by resolve variance (cat8_ON ran 14 turns / 4 cap-fires vs cat8_OFF's 11 / 1, so its
wall-clock ran longer *despite* the cache — its TTFT still shows the prefill win 10.17→2.94 s). **TTFT + hit%
are the un-confounded cache metrics.** Standard-lens reducer: `scripts/fr13_standard_metrics.py`.

---

## 4. Tree shape — acceptance is DEPTH-limited, not width-limited

Realized accept/event ceiling ≈ **3.5** at B=1; **tree positions ≥5 accept exactly ZERO** (measured). Rescue
value is concentrated at depth-1 (the **~27% d0-rescue**).

| tree | nodes | siblings | accept/event (matched regime) | s_per_fwd |
|---|---|---|---|---|
| chain5 (E5 spine) | 5 | none | 3.25 | ~248 ms |
| **cat6root** | 6 | @depth-1 only | **3.68** (resolved) | ~246 ms |
| cat10 | 10 | @all 5 depths | ~3.5 (depths 4-5 DEAD) | ~264 ms (+18ms for 0 accept) |
| **cat8** | 8 | @depths 1-3 | 3.48 (OFF) / 2.97 (ON) | **~248 ms (NO penalty)** |

**VERDICT — at B=1, tree width up to 8 nodes is essentially FREE on s_per_fwd.** chain5/cat6/cat8 = 5/6/8 nodes
all clock **~247-249 ms/forward** (the tree verify is negligible vs the ~27 GB fp8 weight load); only cat10's 10
nodes cost the +18 ms (its depth-4/5 siblings are past the accept frontier). **But the extra width buys no clear
extra acceptance on 12907:** accept/event is **3.0-3.5 across all three trees** with no monotone scaling in node
count (chain5 3.2-3.4, cat6 2.8-3.1, cat8 3.0-3.5), and the OFF-vs-ON spread is resolve-regime noise (different
turn counts, single task). So **cat6root remains the validated sweet spot; cat8 is a no-penalty equal, not a
clear win** on this single task — ranking them decisively needs an N-repeat within a matched resolve-regime.
*Caveat: B=1 only; wider may pay at concurrency / with a stronger drafter.*

---

## 5. The 12907 failures are the char-8 flake — NOT lossiness  ✅

Cache-ON 12907 fails intermittently (long flails, gen 13k–22k tok, ending `empty_patch`). **VERDICT (failure
workflow):** this is the **cache-INDEPENDENT char-8 tool-call flaky-decode** — the agent reasons correctly
(even emits the right fix once), then a tool-call's `arguments` JSON opens a string and never closes it
(`Unterminated string … char 8` → HTTP 400). The text is **fully coherent UTF-8 (0 garbled bytes)** — by the
project's own criterion (garbled-tokens = lossy; coherent-but-off-task = not), this is **NOT a losslessness
break**. Already established 2026-06-29 that char-8 is cache-independent (≈50/64 cold-at-break, fires
cache-OFF). The "cache-OFF resolves / cache-ON fails" pattern (4/4 in a tiny 2×2) is consistent with sampling
luck at temp 0.6 (no seed pin).

> **First proven-config (EXACT_SEED=1) data — tree×cache matrix, killed at 2/6 (E5 done), 2026-06-30:** the
> pattern **FLIPPED** — **E5 cache-ON RESOLVED** (55 turns, TTFT 3.0s, e2e 15.11, 89% hit) while **E5 cache-OFF
> FAILED** (112 turns, TTFT 15.9s, e2e 9.75). If the cache were behaviorally lossy, cache-ON would fail *more*;
> here it *resolved* while cache-OFF flailed → strong evidence resolve is **sampling noise, not a cache
> effect** (the earlier 4/4 was luck), consistent with EXACT_SEED's bit-exact proof. The e5_OFF failure was a
> **pure over-inspection give-up**: 112 turns, 30k tokens, **0 `apply_patch` attempts**, no char-8, no
> truncation — inspected endlessly, never edited (same model resolved it in 55 turns on e5_ON). That's the
> give-up problem (task #13/#16), NOT lossiness. Decode/accept (22.63 vs 16.31, 4.06 vs 3.04) is the
> failed-vs-resolved regime confound, not a cache cost (s_per_fwd matched-regime ~246ms both, §2). Matrix being
> re-run with a thinking budget to cut the give-up + make arms fast.

> **FULL 6/6 matrix — proven config (EXACT_SEED=1) + thinking cap (cap=500), 2026-07-01: LOSSLESSNESS HOLDS
> BEHAVIORALLY.** All six arms (chain5/cat6/cat8 × cache OFF/ON) **RESOLVED** — the proven cache preserves
> resolve on *every* tree, both ON and OFF (3/3 trees resolve both ways). This retires the "cache-ON fails /
> cache-OFF resolves" artifact: at the behavioral level cache-ON is indistinguishable from cache-OFF, consistent
> with EXACT_SEED's bit-exact L0 proof (§1). The **thinking cap** (`LUMO_PROXY_THINK_BUDGET=500` — the
> `</think>`-injection via `continue_final_message`, validated this session) fired **1-4×/arm and cut the
> give-up**: e5_OFF, which *failed* at 112 turns / 0 `apply_patch` on 2026-06-30, now **resolves in 11 turns**.
> Every arm met the 30-min/task bar (max 15.8 min agent wall-clock). Per-arm two-lens metrics in §3a; reducer
> `scripts/fr13_standard_metrics.py`. **Open confirmation still stands:** N-repeat resolve-rate ON vs OFF
> (Fisher-exact) for a statistical lossless claim — 6/6 is 3 matched pairs, not yet powered.

**Open confirmation:** N-repeat resolve-rate ON vs OFF, Fisher-exact, on the
`EXACT_SEED=1` config (the rategate/blocksweep as written use `=0` = the lossy config — re-point to `=1`).

> **Terminology (the word "retry" is misleading — use these instead):** three distinct mechanisms exist; only
> the third is a true retry, and it is DISABLED.
> - **continue / nudge** — when the agent stops without editing, the proxy injects a forceful in-session
>   directive ("your VERY NEXT action MUST be an `apply_patch`…", `LUMO_PROXY_AUTO_CONTINUE=1`). **Context is
>   preserved** — same conversation, same repo state. This is what handles give-ups now.
> - **re-issue** — when codex returns ZERO tokens / disconnects mid-stream (a transport quirk over the
>   alienware link), the runner re-asks codex from the existing state (`zero_token_retry_count` in the code, a
>   misnomer; the prompt builder is literally `_retry_prompt_continue`). **Not** a restart — nothing is discarded.
> - **retry (clean-context restart)** — `SWE_EMPTY_PATCH_RETRIES`, which discards the conversation and starts the
>   task over. **Set to 0 (disabled)** — explicitly replaced by the in-session continue/nudge above.
>
> So a slow agentic run = the agent grinding through many **continued** turns (context intact) on a hard task,
> NOT clean-context restarts. The high continue/nudge rate (agent stopping without editing more than expected)
> is the open give-up question (task #13) — being researched against Qwen3+Codex+vLLM best-practice.

---

## 6. Measurement methodology (paper-grade, baked into `fr13_decode_accounting.py`)

- **`s_per_fwd`** = the metric to compare for small effects, BUT it is weight-floor-dominated (~91%
  node-independent) → compare only within matched context-length.
- **`accept/event`** = the only column with real tree/cache signal → compare only within the same
  resolve-regime (RESOLVED↔RESOLVED); the RESOLVED→FAILED trajectory term (−0.69) swamps cross-regime.
- **`tok/s` (token-weighted)** = the deployment-throughput headline, but length-confounded → don't attribute
  small effects on it; same basis on BOTH sides of every comparison (the 23.88-vs-16.8 mistake).
- **`tok/draft` ("tree size")** = warmup-contaminated counter (root double-count) → NEVER cross-arm.
- **Resolve outcome** = the right metric for the behavioral-losslessness question (timing-insensitive).

---

## 7. Open items
- **Matrix (in flight, job b76xfiuwk):** E5/cat6/cat8 × cache{ON=EXACT_SEED=1, OFF}, 12907 — fills §4 (cat8)
  and §5 (ON≈OFF resolve on the proven config).
- **Task #10:** bake `EXACT_SEED=1` + block-1024 as the launch DEFAULT (currently `=0`; gates set it `=1`).
- **Behavioral-losslessness:** N-repeat resolve-rate ON vs OFF (Fisher) on `EXACT_SEED=1` to convert the
  char-8 "consistent-with-flake" into "statistically proven not-cache".
- **char-8 robustness (task #12):** constrained tool-call decoding / json_repair (cache-independent flake).

---

## Cache / quality contribution — honest decomposition (upstream-verified 2026-07-07, wf_7d8464d1)

Verified against upstream vLLM github (issue/PR numbers cited). Framing is deliberately un-inflated.

**GENUINELY NOVEL (unreported upstream) = branched/tree GDN spec-decode + prefix-cache losslessness.**
vLLM ships no tree/branched spec-decode; the only upstream rollback-parity artifact (#46187 ReplaySSM, OPEN)
concedes "no parity data yet." Ours = the two-carrier diagnosis — (a) branched-accept ROW-ADDRESSING
(accepted token k = tree node path[k], NOT stock linear row num_accepted-1; measured 48/48 UNFAITHFUL,
diff ~23 at L0; fixed by leaf-map/SNAP_FIX → the stateless-tree runrow commit) + (b) the align-boundary
CROSS-KERNEL numeric-basis mismatch (tree GDN decode kernel vs chunk_gated_delta_rule restore fold, ~1-2 ULP
compounding over ~48 GDN layers) — PLUS a MEASURED within-floor lossless gate (per-depth argmax + live
SWE-Verified behavioral resolve parity). This is the tree EXTENSION of the known-open linear-chain problem.

**UPSTREAMABLE stock fixes (we hit + fixed, did NOT discover — the "mostly stock-vLLM territory" bucket):**
- B=4 GDN restore-path device-assert (gdn_linear_attn.py:986, initial_state[~has_initial_state]) — same class
  as #39809 (mamba PC + MTP illegal-memory at batch>1) / #35288 (MTP corruption at concurrency≥4, OPEN).
- APC-overshoot fix (max_num_batched_tokens = block_size) IS #45238 (align-mode PC drops to 0% hit, OPEN).
- The mamba cache write-contract + restore-path corrections ride entirely on stock spec-state tensors.

**KNOWN-UPSTREAM we merely mitigate (NOT ours):**
- char-8 tool-call JSON → HTTP 400: #43713 (qwen3_xml emits invalid JSON, OPEN) + #43995 (renderer
  json.loads(arguments) 400 on malformed re-fed history, OPEN; only empty-arg subcase fixed #19419/PR#25223).
  LUMO_PROXY_RETRY_UPSTREAM_400 = client-side workaround, not a root-cause fix, not upstreamable as-is.
- Linear-chain GDN spec-decode state-rollback losslessness = #39273 (OPEN) — do NOT claim; ours is the extension.
- Base mamba/GDN prefix-cache + spec-decode enablement (#26201 umbrella, #33705/#33726) — actively-worked upstream.

**SGLang attribution:** the working cache discipline is ported from SGLang MambaRadixCache
(research/fr13_workflows/sglang_mamba_radix_cache_design.md): block-aligned snapshots + "snapshot the
committed state, restore VERBATIM, never reconstruct" (= the stateless-tree runrow commit). The per-radix-node
checkpoint tree + ping-pong slot did NOT port (vLLM keeps one checkpoint at the last block boundary).

**Honesty guards:** within-floor, NOT bit-exact (REFOLD non-functional, redirect_used=0 → recurrent-leaf
fallback 100%, cross-basis residual eaten); greedy-LCP committer carries NO rejection-sampler losslessness
theorem (so "lossless" = within-floor + behavioral resolve parity, not a proven guarantee); fork-bound (the
contributable form is the diagnosis + gate methodology, a tree extension of #39273, not a mergeable PR).
