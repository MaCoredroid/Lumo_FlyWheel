# FR13 Serving Memory-Leak: Root Cause + Fix (adversarially verified)

Date: 2026-07-01
Container inspected LIVE: `fr13-bigdenom-m_e5_ON` (EngineCore pid=176), up ~34 min,
`FR13_APC_EXACT_SEED=1`, `MAX_NUM_SEQS=4`.

---

## (1) LOCUS — HOST RSS, not the GPU KV pool  [CONFIRMED]

- GPU KV pool is FLAT. Live `/metrics` and the historical artifacts
  `.../per_task/astropy__astropy-12907/vllm_metrics_{pre,post}.txt` both read
  `vllm:kv_cache_usage_perc = 0.0`. The pool is FIXED-size (vLLM v1 `BlockPool`
  allocates `num_gpu_blocks` once at init from GPU-util-derived `available_memory`,
  `kv_cache_utils.py:860`; no grow path — `get_new_blocks()` just raises when
  exhausted). So `kv_usage` is a fraction of a constant denominator → it CANNOT be
  the leak.
- The growth lives in HOST RSS: docker `MemUsage` 13.81 GiB ≈ cgroup
  `memory.current` 13.86 GiB (they agree → container-resident host memory,
  separate from the static ~79.8 GiB GPU allocation). Empirical active-window
  growth 11.8 → 13.5 GiB over 20 min = **~87 MiB/min**.
- MECHANISM tying the accumulating structure to host RSS: the checkpoint tensors
  are stored with `.cpu()` (see write site below), i.e. **CPU tensors → host RSS**,
  exactly the observed locus.

Idle caveat (honest): my own live 3× sampling was during an IDLE server
(`num_requests_running=0`, `prefix_cache_queries_total` frozen at 1138467), so RSS
was flat (13.81 GiB) and I could NOT observe live growth. The 87 MiB/min is
workload-correlated, quantified only from the known active window, not re-measured
live.

---

## (2) THE ACCUMULATING STRUCTURE

There are TWO distinct FR13 stores. The sub-agents disagreed on which is the leak;
LIVE LOG EVIDENCE resolves it.

### PRIMARY driver: `BlockPool._fr13_es_ckpt`  (block-hash-keyed, CPU tensors)
- Keyed by: `block_hash` (`BlockHashWithGroupId`).
- Value: `{'pos': int, 'layers': {layer_str -> CPU float32 tensor}}`. Each layer
  tensor is the GDN SSM state `[num_v_heads=48, head_k=128, head_v=128]` f32 =
  **3.0 MiB/layer**; a fully-populated entry = 48 layers = **144 MiB**.
- Written by `_fr13_es_try_bind` (`store[h] = {'pos', 'layers': layers}`) via the
  `ES_WRITE` path; the stored `layers` dict is the SAME object as
  `_FR13_ES_PENDING_BY_REQ[req][pos]`, and later per-layer captures keep filling it
  (observed `nlayers` climbing 1→48).
- Pruned ONLY by `BlockPool._maybe_evict_cached_block` (block eviction) or
  `reset_prefix_cache`. **Live evidence: `docker logs | grep -ci evict` = 0.** With
  the KV pool at 0% there is no memory pressure, so blocks are essentially never
  evicted → this store is pruned essentially never during a run.
- LIVE COUNTS: `ES_WRITE` fired 4992× over 05:26→05:43 (~17 min); **38 distinct
  block-hashes**, all `nlayers` 1..48 each seen 104×. Ceiling = 38 × 48 × 3 MiB =
  **~5.3 GiB** of host-RSS CPU tensors.
- WHY it isn't pruned: the request-finish free-hook DOES pop
  `_FR13_ES_PENDING_BY_REQ` by req_id, but the block store holds an independent ref
  to the same `layers` dict, so freeing the request does NOT release it — by design
  it must outlive the request for cross-turn restore. Its ONLY reaper is
  block-eviction, which does not fire.

### SECONDARY (bounded, NOT the per-minute leak): the three per-GDN-layer slot dicts
`self._fr13_apc_chunked_ckpt_by_req`, `self._fr13_apc_pending_kvab`,
`self._fr13_apc_abs_pos_by_req`.
- All three sub-agents flagged these as the unbounded leak. **That claim is
  REFUTED by their own key.** They are written keyed by the physical SSM STATE
  SLOT string `str(int(non_spec_state_indices_tensor[r]))` (line ~6264), and the
  free-hook (`_patch_scheduler_fr13_freereq_cleanup`, line ~7246) pops by
  `(request_id, str(request_id))` — key namespaces never match, so the free-hook is
  INERT for them (correctly diagnosed). BUT slot-keying is exactly what BOUNDS them:
  the SSM state-slot pool is sized by `max_num_seqs` (=4) plus a small spec margin.
  **Live evidence: only 6 distinct slots ever appear (1,7,13,20,26,32), each written
  16× with only 2 distinct req-ids; each re-write OVERWRITES the same key.** So the
  dict size is capped at ~6 entries; high-water ≈ 6 × 144 MiB ckpt + 6 × ~47 MiB
  pend ≈ **~1.1 GiB ONE-TIME**, not a per-minute leak. (These are also GPU tensors
  `final0[0]...clone()`, i.e. NOT the host-RSS locus.)

Net: the free-hook key-mismatch is a real bug, but it is a bounded ~1 GiB GPU
high-water, not the 87 MiB/min host-RSS growth. The host-RSS driver is the
block-hash CPU-tensor ckpt store with no firing eviction reaper.

---

## (3) CACHE-ON-SPECIFIC?  YES — and it matters for the plan.

- The entire capture/`ES_WRITE`/ckpt-store path is gated on
  `FR13_APC_EXACT_SEED == "1"` (line ~6148); the block-hash store keys off prefix
  cache block hashes (only exist with prefix caching = cache ON).
- Therefore **cat8_OFF (cache-OFF) arms will NOT populate this store and will NOT
  exhibit this leak.** The user's plan running cat8_OFF is safe from this specific
  leak. The leak is a cache-ON / EXACT_SEED-ON artifact.

---

## (4) THE PRECISE MINIMAL FIX

Two independent defects; fix both, the first is the host-RSS driver.

**FIX A (primary — bound the block-hash CPU-tensor store):** give
`BlockPool._fr13_es_ckpt` an eviction reaper that actually fires under this
workload. The canonical evict funnel already exists —
`BlockPool._maybe_evict_cached_block(self, block)` pops the store — but it never
runs because there is no memory pressure. Add an independent CAP so the store is
bounded regardless of eviction:
  - Make `_fr13_es_ckpt` an LRU (`OrderedDict`) with a hard entry cap
    (e.g. `num_gpu_blocks`, or a modest constant like 256). On insert in
    `_fr13_es_try_bind`, if over cap, `popitem(last=False)` the oldest entry
    (dropping its `layers` CPU tensors). This bounds host RSS to
    cap × 144 MiB and cannot grow with cumulative-distinct-hashes-ever-seen.
  - This is the ONE structure whose growth is otherwise unbounded across a long
    multi-task SWE run (each new task → new prefix blocks → new block-hashes; with
    eviction never firing, distinct-hashes-ever accumulates).

**FIX B (secondary — make the free-hook actually reap the slot dicts):** the
per-GDN-layer dicts are pruned by the free-hook only if keys match. Either
  - key the three dicts by req_id (revert the GAP-a slot re-key), OR
  - at request-free, map req_id → slot and pop by slot, OR
  - restore a per-turn slot pop.
This only recovers ~1 GiB of GPU high-water; it is NOT what stops the host-RSS
growth. Do it for correctness/tidiness, not as the RSS fix.

---

## ADVERSARIAL RATE CHECK — honest verdict

- Store ceiling 38 hashes × 144 MiB = **5.34 GiB**; ES_WRITE window ~17 min ⇒
  average fill ~**322 MiB/min**. This OVERSHOOTS the empirical 87 MiB/min by ~3.7×.
- Reconciliation (not hand-waving): (a) the 87 MiB/min was measured over a
  DIFFERENT 20-min window than the 17-min ES_WRITE burst; fill is front-loaded (new
  distinct hashes appear early, then re-writes just overwrite), so instantaneous
  rate late in a task is far below the 322 average; (b) `.cpu()` tensors may share
  pages / be partially reclaimed; (c) the 87 vs the task's stated ~113 MiB/min
  already shows the rate estimate is soft. The direction and locus match (host RSS,
  CPU tensors, cache-ON-gated); the exact coefficient does NOT tightly match.
- Order-of-magnitude: PLAUSIBLE and correct-sign. Tight numeric match: NOT proven.

### The ONE measurement that would NAIL it
Set `FR13_LEAK_PROBE=1` (already wired to print `pend_kvab`/`ckpt` sizes every 25
frees) AND add a one-line print of `len(block_pool._fr13_es_ckpt)` +
`sum(len(v['layers']) for v in ...)` on the same cadence, then drive ONE active SWE
task while sampling cgroup `memory.current` every 10 s. Correlating
`d(RSS)/dt` against `d(ckpt_store_bytes)/dt` in the SAME active window (not idle,
not a different historical window) is the single measurement that converts this from
"correct locus + plausible driver" to "quantitatively pinned."

---

## CONFIDENCE

- Locus = host RSS, not KV pool: **HIGH** (live + artifact metrics both 0.0; cgroup
  vs GPU allocation cleanly separated).
- Cache-ON / EXACT_SEED-specific ⇒ cat8_OFF safe: **HIGH** (explicit env gate at
  line ~6148; store keyed on prefix-cache block hashes).
- PRIMARY driver = block-hash `_fr13_es_ckpt` CPU-tensor store with no firing evict
  reaper: **MEDIUM-HIGH** (right locus = CPU tensors → host RSS; live evidence of
  4992 writes / 38 hashes / nlayers→48 / 0 evictions; but the exact MiB/min
  coefficient overshoots, so not fully pinned).
- Slot-keyed per-layer dicts are the unbounded leak (the sub-agents' claim):
  **REFUTED** — bounded to ~6 slots by construction (live: 6 slots, overwrite-in-
  place), ~1 GiB GPU one-time high-water, and GPU-not-host so wrong locus anyway.
  The free-hook key-mismatch is real but is a tidiness bug, not the RSS driver.
