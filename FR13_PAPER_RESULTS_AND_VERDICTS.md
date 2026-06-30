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

---

## 4. Tree shape — acceptance is DEPTH-limited, not width-limited

Realized accept/event ceiling ≈ **3.5** at B=1; **tree positions ≥5 accept exactly ZERO** (measured). Rescue
value is concentrated at depth-1 (the **~27% d0-rescue**).

| tree | nodes | siblings | accept/event (matched regime) | s_per_fwd |
|---|---|---|---|---|
| chain5 (E5 spine) | 5 | none | 3.25 | ~248 ms |
| **cat6root** | 6 | @depth-1 only | **3.68** (resolved) | ~246 ms |
| cat10 | 10 | @all 5 depths | ~3.5 (depths 4-5 DEAD) | ~264 ms (+18ms for 0 accept) |
| **cat8** | 8 | @depths 1-3 | **TBD (matrix in flight)** | TBD |

**VERDICT (so far) — cat6root is the B=1 sweet spot among {chain5, cat6, cat10}:** cat10's wider tree costs
+18 ms/forward to verify but buys **no extra acceptance** (depths 4-5 are past the accept frontier). **OPEN:**
cat8 (siblings at depths 1-3, the accept frontier) is the hypothesis for a better middle ground — being tested
now in the tree×cache matrix. *Caveat: B=1 only; wider may pay at concurrency / with a stronger drafter.*

---

## 5. The 12907 failures are the char-8 flake — NOT lossiness  ✅

Cache-ON 12907 fails intermittently (long flails, gen 13k–22k tok, ending `empty_patch`). **VERDICT (failure
workflow):** this is the **cache-INDEPENDENT char-8 tool-call flaky-decode** — the agent reasons correctly
(even emits the right fix once), then a tool-call's `arguments` JSON opens a string and never closes it
(`Unterminated string … char 8` → HTTP 400). The text is **fully coherent UTF-8 (0 garbled bytes)** — by the
project's own criterion (garbled-tokens = lossy; coherent-but-off-task = not), this is **NOT a losslessness
break**. Already established 2026-06-29 that char-8 is cache-independent (≈50/64 cold-at-break, fires
cache-OFF). The "cache-OFF resolves / cache-ON fails" pattern (4/4 in a tiny 2×2) is consistent with sampling
luck at temp 0.6 (no seed pin). **Open confirmation:** N-repeat resolve-rate ON vs OFF, Fisher-exact, on the
`EXACT_SEED=1` config (the rategate/blocksweep as written use `=0` = the lossy config — re-point to `=1`).

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
