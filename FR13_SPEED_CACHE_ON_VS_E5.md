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
| **cat6root cache-ON** | EXACT_SEED + 1024 + full graph | **17.9** (1-task, leak-fixed) → _4-task pending_ | **16.3** | 59% | the deployed lossless config; `.clone()`-taxed |
| **cat10 cache-ON** | EXACT_SEED + 1024 + full graph | **17.2** (1-task, leak-fixed) | **14.5** | 35% | 10-node tree: more drafts but lower accept → **≈ cat6root decode at B=1** (no win) |
| cat6root cache-OFF | FIX-1/2/3, no EXACT_SEED, prior | **23.88** (4-task) | _+4.0% over E5 (latency)_ | — | the +27% decode run; e2e floods (no cache) |
| **E5 / spine-5** (chain5) cache-OFF | native 5-step MTP spine | **18.80** (4-task) | baseline | — | the spine baseline cat6root beats by +27% decode |

> **All cache-ON numbers above are 1-task (12907) leak-fixed** — the clean 4-task token-weighted run is
> blocked by a residual ~0.4 GiB/min leak (the committer-prune bounded the dominant leak; the fixed-buffer
> port — task #15 — caps the residual + recovers the `.clone()` decode tax). decode is per-token so the
> 1-task figure is a good estimate; the 4-task average will be close, and the fixed-buffer moves it *up*
> toward 23.88 by removing the EXACT_SEED clone overhead. **cat10 ≈ cat6root at B=1** — the wider tree's
> extra drafts don't pay off at batch-1; cat6root's 6-node root-branch is the better shape here.

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

**Status:** leak fixed + verifying (v3); cache-ON 4-task numbers + the clean 3-way to be filled in here.
