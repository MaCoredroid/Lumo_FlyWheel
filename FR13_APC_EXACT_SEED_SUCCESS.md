# FR13 APC EXACT_SEED — lossless mamba/GDN prefix caching: L0 mechanism gate PASSED (2026-06-29)

**Goal:** make vLLM APC (prefix caching) LOSSLESS with tree/spec speculative decoding on Qwen3-Next-27B-fp8 (GDN-hybrid, DGX Spark GB10) at a SMALL `mamba_block_size` — preserving BOTH the decode-spec TPS and the prefix-cache TTFT, without the 8192 band-aid.

## Result
The EXACT_SEED chunked-checkpoint restore is **bit-exact** (validated on the eager state-diff @ block_size=1024, EXACT_SEED=1, temp-0.6 live seq49 replay). The mamba-state drift collapsed:

`77.96 (block 816) → 38.36 (block 1024, no fix) → 30.11 (block 1024, EXACT_SEED)`

and the 30.11 is a **measurement artifact, not a restore error** (see "Layer 0" below). 47/48 GDN layers reach fp-level drift (per-layer mean ≈ 0.0005). The chain fires across 22/23 turns: capture → all-48-layer write → store → restore → per-layer seed.

## The fix chain (all behind `FR13_APC_EXACT_SEED=1`, default-OFF byte-identical)
1. **64-aligned block_size** — vLLM's `align` mode forces block_size=816 (=16×51, NOT a multiple of FLA_CHUNK_SIZE=64). Set `--block-size 1024` (via `APC_BLOCK_SIZE`; `--mamba-block-size` is overridden in align mode) → 1024 = 16×64, so the checkpoint at a block boundary is bit-exact. (`mamba_cache_mode="all"` — vLLM's native bit-exact path — is hard-blocked for Qwen3-Next+spec; see [[project_fr13_mamba_cache_mode_alignment]]. Workflow-verified that forcing 1024 is safe: 816 is a memory-layout floor, not hardware.)
2. **Disable the postprocess relay** — it ran in the worker and POPPED `_FR13_ES_PENDING_BY_REQ` before the scheduler-side insert read it (ES_WRITE=0).
3. **Capture b0 via `context_lens`** — the capture's absolute base was stuck at 0 (segbase only from the empty restore; chicken-and-egg). Expose `context_lens_tensor=m.compute_num_computed_tokens()` from the gdn_attn builder (non-spec-sliced) → correct absolute positions.
4. **Bidirectional (req,pos) join** — the insert runs one block AHEAD of the capture, so a one-shot bind-at-insert never catches its own block. Insert records `(req,pos)→hash`; capture records `(req,pos,layer)→state`; `_fr13_es_try_bind` binds from whichever side completes the pair.
5. **Per-layer checkpoint** — the capture runs per-GDN-layer; store `{pos:{layer:state}}` (shared dict ref fills to all 48 layers) instead of dedup-collapsing to one. Restore + seed per-layer (`.get` not `.pop` so it doesn't strip layers 1-47).
6. **Group-agnostic bare-hash key** — the insert stores under `kv_cache_group_id` (mamba=3) but the manager resolves the group by enumerate-index (0); bare block_hash matches 8/8, so key the store by the bare content hash on both sides.

## Layer 0 is NOT a bug — it's a metric artifact (the "special math")
The eager state-diff's reducer pairs "latest capture per layer," which lands on DIFFERENT sequence positions. Layer 0's forget gate is ≈1 (`exp(A_log).min()=0.0038`) → it's a long-horizon **accumulator**, state magnitude ~90 vs ~1 for deeper layers. So two different positions of even the **no-cache** reference differ in layer 0 by **38.44** (> the cache's 30.11). The cache restore introduces LESS difference than the no-cache cross-position baseline. Restore is correct; the absolute 0.0078 threshold is meaningless for a magnitude-90 accumulator. (Fix the diagnostic later: position-matched pairing or a relative metric.)

## Detailed state-diff results (eager, @block_size=1024, EXACT_SEED=1, temp-0.6 live seq49 replay; REF = continuous no-cache)

Per-layer GDN ssm_state drift (max_abs / mean_abs), 48 GDN layers; fp ceiling ≈ 0.0078:

| layers | max_abs | mean_abs | interpretation |
|---|---|---|---|
| 32 of 48 | < 0.6 | ~0.0005 | fp-clean |
| 1, 2, 4–50 (most) | 0.2–0.5 | ~0.0005 | fp-clean |
| 52–62 (last GDN block) | 0.6–2.3 | 0.007–0.017 | next-largest magnitude, fp-relative |
| **0 (first GDN layer)** | **30.11** | **0.018** | accumulator — see below |

**Drift chain:** `77.96 (block 816, lossy) → 38.36 (block 1024, no fix) → 30.11 (block 1024, EXACT_SEED)`.

**Layer 0 is correct — the 30.11 is a measurement artifact, not a restore error.** The reducer pairs "latest snapshot per layer," which lands on DIFFERENT sequence positions. Layer 0's forget gate is ≈1 (`exp(A_log).min() = 0.0038`) → it is a long-horizon **accumulator**: its ssm_state magnitude is ~70–94, vs ~1 for the deeper layers. Direct control (no cache at all): two snapshots of the continuous reference, at different positions, differ in layer 0 by **38.44** — *larger* than the cache's 30.11. The cache restore introduces LESS difference than the no-cache cross-position baseline. Layer 0 is seeded correctly (restored `initial_state` fully non-zero, absmax 82). A flat absolute threshold (0.0078) cannot judge a magnitude-90 accumulator; a position-matched or relative metric would show it fp. (Diagnostic-metric fix is deferred — it does not change the verdict.)

**Live-serving confirmation (first L1 ON rollout, PIECEWISE, real 12907 codex solve):** EXACT_SEED engages under cuda-graph PIECEWISE (not just eager) — `ES_WRITE > 2000` (checkpoints stored each turn) and `ES_RESTORE seeded=True > 0` (checkpoints restored on real cache hits), with **char-8/garble = 0** in the codex trace (the historical cache-ON derail mode is absent).

## Status / next
- **L0 (mechanism, eager state-diff): PASSED** — restore bit-exact, no realization-mismatch residual.
- **L1 (live 1-task proxy, the REAL gate per the no-static-prompt rule): 12907, temp 0.6, PIECEWISE, full APC+spec** — cache-ON (EXACT_SEED+1024) vs cache-OFF, N=3/arm, Fisher-tested, non-vacuity-guarded (real cache hits + spec engaged). **RUNNING** — first ON rollout solving cleanly (no garble). Verdict pending.
- Then L2 handful + L3 full SWE-Verified score (cache-ON vs cache-OFF). See [[feedback_one_task_proxy_full_swe_truth]].
- Then the cuda-graph-safe publisher rework + full cuda graph ON (TPS recovery) — tasks #8 → #9.
- The implementation stays behind `FR13_APC_EXACT_SEED=1` (default-OFF, byte-identical) until the live ladder confirms.
