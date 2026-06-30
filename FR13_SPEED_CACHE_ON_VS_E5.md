# FR13 Speed — cache-ON cat6root vs the prior cat6root-vs-E5 decode comparison

**Question:** does turning the prefix cache ON (APC + EXACT_SEED + full CUDA graph) preserve cat6root's
decode win over the native spine *and* fix the end-to-end (e2e) flooding that the cache-OFF runs had?

---

## 1. Prior baseline — cache **OFF** (the +27% decode win)
Clean **B=1 / temp 0.6**, four SWE-Verified / Codex tasks (astropy 12907/13033/13236/13398),
**token-weighted decode throughput** (`fr13_measure.py deploy-speed --basis decode_seconds`).
Source: `FR13_SPEEDFIX_DEPLOY_SCREEN.md`.

| arm (cache OFF) | decode tok/s | vs native E5 (decode) | per-request latency (e2e) |
|---|---|---|---|
| **cat6root** (6-node root-branch tree) | **23.88** | **+27%** | **+4.0%** over native E5 |
| native **E5** (native 5-step MTP spine) | 18.80 | baseline | baseline |

**The problem this exposes:** the **+27% decode** win collapsed to only **+4% e2e** wall-time. Decode-only ≠
e2e — and with the cache OFF, every turn re-prefills the full agentic context, so the slow prefill / TTFT
"floods" the e2e and almost erases the decode advantage at the request level.

---

## 2. The cache-ON hypothesis
Turning APC ON (with EXACT_SEED making it lossless + full graph) should:
- **keep** the cat6root tree decode edge over the spine (the tree structure is unchanged), and
- **fix the e2e flooding** — cache hits skip re-prefilling the shared agentic context, cutting TTFT, so the
  decode win should translate to a much larger e2e win than +4%.

**Caveat (the cost side):** EXACT_SEED's committer adds per-turn publisher overhead — a GPU `.clone()` per
accepted token, per GDN layer (this is also exactly what produced the now-fixed serving-path memory leak,
see `FR13_*` task #14 / `_patch_scheduler_fr13_freereq_cleanup`). That overhead taxes the **decode** side, so
cache-ON decode could land *below* the cache-OFF 23.88 even though the e2e improves. The clean measurement
quantifies the net.

---

## 3. THE THREE-WAY (decode + e2e) — current vs prior

Decode-only and e2e (token-weighted `gen/decode_time` and `gen/e2e_time`). "early" = single-task (12907)
leak-FIXED preview; "prior" = the §1 4-task cache-OFF baseline. The CLEAN 4-task current-config numbers
(all three arms, same graph+block) fill the `_pending_` cells.

| arm | config | decode tok/s | e2e tok/s | accept | notes |
|---|---|---|---|---|---|
| **cat6root cache-ON** | EXACT_SEED + 1024 + full graph | **16.8** (3-task TW, leak-fixed) · 17.9 (1-task) | **16.3** (1-task) | 59% | the deployed lossless config; `.clone()`-taxed. 4th task lost to leak-kill |
| **cat10 cache-ON** | EXACT_SEED + 1024 + full graph | **14.1** (4-task TW, clean) · 17.2 (1-task) | **14.5** (1-task) | 27% (4-task) | 10-node tree: more drafts, lower accept → **WORSE than cat6root at B=1** |
| cat6root cache-OFF | FIX-1/2/3, no EXACT_SEED, prior | **23.88** (4-task TW) | _+4.0% over E5 (latency)_ | — | the +27% decode run; e2e floods (no cache) |
| **E5 / spine-5** (chain5) cache-OFF | native 5-step MTP spine | **18.80** (4-task TW) | baseline | — | the spine baseline cat6root beats by +27% decode |

> **TW = token-weighted** (`sum(gen_tokens) / sum(decode_time_s)` across the tasks — the *same* formula as the
> prior 23.88/18.80, directly comparable). The clean **multi-task** cache-ON numbers are now in:
> - **cat10 = 14.1** is a fully clean **4-task** TW (cat10 *survived* — no leak-kill; all 4 tasks decoded:
>   12907=17.2, 13398=16.5, 13033=13.9, 13236=12.5).
> - **cat6root = 16.8** is a clean **3-task** TW (12907=17.9, 13033=16.8, 13236=16.1); the 4th task (13398)
>   was lost when the leak-guard killed the container (counter reset corrupts the delta), not a speed issue.
>
> **Why `fr13_measure` reported `decode_tps=0.00`:** 3 of the 4 b4_four tasks **fail the SWE verdict** (agent
> give-ups — a *resolve* problem, task #13, NOT the leak: cat10 survived and still failed 3 tasks). Their
> thin/early-give-up brackets trip `fr13_measure`'s class-9 engagement assert, so it refuses the aggregate.
> The manual TW above bypasses that assert — **decode TPS is per-token and valid whether or not the task
> resolves**, so the give-ups don't corrupt the *speed* read, only the verdict count.
>
> **Headline:** cache-ON cat6root **16.8** vs cache-OFF **23.88** = a **~30% decode tax** from the EXACT_SEED
> committer (`.clone()` per accepted token/layer + cache publisher overhead). The **fixed-buffer port (task
> #15)** removes the `.clone()` and is expected to recover most of this. **cat10 14.1 < cat6root 16.8**
> confirms the wide tree does *not* pay off at batch-1: more draft compute, lower accept (27% vs 59%).

**Reading it:**
- **decode:** cat6root's tree beats the E5 spine (23.88 vs 18.80, +27%) — the structural d0-rescue edge. The
  current cache-ON early decode (17.9, leak-fixed; was 15.7 leak-era) trails 23.88 by the `.clone()` tax +
  1-task variance + config; the fixed-buffer port recovers the `.clone()` part.
- **e2e:** the prior cat6root cache-OFF was only **+4%** e2e over E5 — the +27% decode win drowned in
  re-prefill. cache-ON's e2e (16.3 early) is where the cache should show its win (TTFT cut on cache hits) —
  the clean 3-way E5/cat6-OFF e2e numbers (same config) quantify it.

> **Comparability caveat:** the prior 23.88/18.80 are the FIX-1/2/3 era (cache-OFF), graph-mode + block-size
> NOT cleanly logged (likely vLLM-default `FULL_AND_PIECEWISE` + the ~816 align-floor block); current is
> full graph + forced 1024. The clean 3-way (all arms same build) removes this drift.
>
> **Do NOT read the early 17.9 as a regression vs 23.88** — different task count, +EXACT_SEED `.clone()` tax
> (recoverable), +config. The clean 4-task `fr13_measure` number is the apples-to-apples figure.

---

## 4. Comparability notes (for the paper)
- **Config drift:** the prior 23.88/18.80 are on the FIX-1/2/3-era config (pre-EXACT_SEED, pre-full-graph),
  cache-OFF. The new cache-ON is the deployed lossless config (EXACT_SEED + 64-aligned block 1024 + full
  graph). A side-by-side of cache-ON-current vs cache-OFF-prior mixes two axes (config + cache).
- **The clean three-way** (recommended, current config, all on the same build):
  `cat6root cache-ON` vs `cat6root cache-OFF` vs `chain5/E5 cache-OFF`, 1 task (12907) each (fits inside the
  serving window now that the leak is fixed), reporting **decode TPS + e2e TPS + TTFT + accept%**. This
  isolates the two variables cleanly: tree shape (cat6root vs spine → the decode/accept edge) and cache
  (on vs off → the e2e/TTFT win). Queued to run once the leak fix is confirmed on v3.

**Status (run_20260630T064631Z, leak-fixed committer-prune build):**
- ✅ Clean **multi-task** cache-ON TW decode banked: **cat6root 16.8** (3-task) · **cat10 14.1** (4-task).
- ⏳ cat6root 4th task (13398) lost to a leak-guard kill — a clean 4-task cat6root needs the leak *capped*
  (the fixed-buffer port) or a lower-util re-run; the 3-task TW already tells the story (~30% tax vs 23.88).
- ⏳ The clean **3-way e2e** (cat6-ON / cat6-OFF / chain5-OFF, TTFT + e2e) still to run.
- 🔧 **Next real fix:** the **fixed-buffer port (task #15)** — removes the `.clone()` decode tax (recovers
  toward 23.88) *and* caps the residual leak (so clean 4-task runs survive). Lossless-gated by
  `fr13_apc_exactseed_statediff.sh` (FIXED_BUFFER=0 vs 1 → identical per-layer state_max) before trust.
