# SWE-Bench Q36-A — concurrency / memory-bandwidth scaling analysis (2026-05-22)

Status: **campaign PAUSED at this point** to digest this before scaling concurrency.
Round 1 (temp10) stopped at 6/55 done; supervise cron deleted; state=paused.

## Correction to the earlier "B=4 → 4×" claim

That was too sloppy. Batching does NOT add bandwidth. We are bound by the
**273 GB/s** memory bandwidth on this DGX (GB10). Batching only changes the
**bytes-per-token ratio** by amortizing the fixed weight fetch across more
output tokens per forward pass. The scaling is **sublinear**, not linear.

Per forward pass we read:
- **Weights**: 27 GB (FIXED regardless of batch size B)
- **KV cache**: B × kv_per_request (scales with batch AND context length)

If weights >> KV (short context) → batching ≈ near-linear.
If weights ≈ KV (long context) → batching sublinear, diminishing returns.

## Scaling table @ ~40K context (KV ≈ 6 GB/request), 273 GB/s

| B | Weights | KV (B×6) | Bytes/pass | Pass @273GB/s | Tok/pass | GB/tok | Aggregate tps | Speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 27 | 6 | 33 | 121 ms | 1 | 33 | 8.3 | 1.0× |
| 2 | 27 | 12 | 39 | 143 ms | 2 | 19.5 | 14.0 | 1.7× |
| 4 | 27 | 24 | 51 | 187 ms | 4 | 12.75 | 21.4 | 2.6× |
| 8 | 27 | 48 | 75 | 275 ms | 8 | 9.4 | 29.1 | 3.5× |
| 16 | 27 | 96 | 123 | 451 ms | 16 | 7.7 | 35.5 | 4.3× |
| 32 | 27 | 192 | 219 | 802 ms | 32 | 6.8 | 39.9 | 4.8× |
| ∞ | 27 | ∞ | ~B×6 | ~B×22 ms | B | 6 | ~45 | ~5.4× ceiling |

**Asymptotic ceiling ≈ 5.4× aggregate at 40K context**, set by KV's 6 GB/tok.
The ratio that governs it is **weights : (B × KV)** — at B=1 it's 27:6 (weights
dominate), at B=4 it's ~1:1 (KV catches up), at B=8 it's 27:48 (KV dominates).
No sharp knee at B=4 — it's a smooth diminishing-returns curve.

## With spec-decode (Q36-A SuffixDecoding)
~0.4 acceptance × 12 draft ≈ ~5 accepted tokens/pass/req → multiply tok/pass by
~3 effective. Helps at every B but does NOT change the bandwidth-bound *shape*:

| B | Aggregate effective tps | Speedup |
|--:|--:|--:|
| 1 | 25 | 1.0× |
| 4 | ~65 | 2.6× |
| 8 | ~85 | 3.4× |
| 16 | ~105 | 4.2× |

## Realistic SWE-Bench (context grows across agent turns)
- Early (T0-T5): 5-15K ctx, KV 1-2 GB → weights dominate → near-linear gains
- Mid (T5-T20): 20-40K ctx, KV 3-6 GB → KV catches weights at B=4-8
- Late (T20+): 40-80K ctx, KV 6-12 GB → KV dominates at B≥3

Time-weighted: **~2-2.5× aggregate at B=4** (not 4×). Full campaign realistic
speedup ≈ **2.5-3×**, reached around B=8-16.

## The real trade-off: per-stream tps drops with B
Per-stream throughput degrades meaningfully past **B≈3-4**. At B=8 per-stream is
~3.6 tps vs ~8 tps single-stream. Codex agents are **wall-budget-bound** (30 min),
so fewer tokens/stream = fewer fix-revise iterations.

Worked example (from temp06 data ~109 model calls/30min at concurrency=1):
- **concurrency=1**: full per-stream (~22 tps), 30 min wall. 500 × 30 min = 250 hr.
- **concurrency=8**: ~12-14 tps/stream, 8 parallel, ~3.5× aggregate → ~88 hr.
  But each agent gets ~38 calls instead of 109 (~65% fewer iterations) → **pass
  rate may drop**.

## Claim audit
| Claim | Verdict |
|---|---|
| Memory bandwidth (273 GB/s) is the ceiling | TRUE |
| Batching doesn't help | FALSE — amortizes weight fetch, sublinear-but-real |
| Concurrency=4 → 4× | FALSE — ~2.5-3× at 40K ctx |
| Per-stream tps doesn't degrade with batching | FALSE — degrades from B≈3 |
| Aggregate plateaus eventually | TRUE — KV dominates; ~5× ceiling |
| Concurrency helps campaign wall time | TRUE, but 2.5-3× not 4×, at cost of per-task iterations |

## Proposed pre-commit experiment (BEFORE scaling campaign concurrency)
Paired 10-instance subset, **concurrency=2 vs concurrency=1** (same instances).
**Codex runs on x86 (alienware) STRICTLY** — DGX GPU dedicated to vLLM batching,
no agent compute on the DGX. Compare:
1. Aggregate wall time (expect ~30-40% faster at B=2)
2. Per-instance pass rate (expect ≈ equivalent if iteration count doesn't bind)
3. Per-stream tps (expect ≈ same at B=2 — KV not yet dominant)

Decision rule:
- If pass rate drops ≥5 absolute points at B=2 → NOT worth it; stay concurrency=1.
- If pass rate invariant at B=2 → push to B=4 and re-test.

Only after this confirms iteration-count is not pass-rate-binding should campaign
concurrency be raised. Until then the paired temperature A/B (and any resumed
Verified/Pro run) stays **concurrency=1**.
