# FR13 — B=4 SWE-Verified E5 TPS Reconcile (history ~40 vs ours 15.66)

Branch: `fr13-speedfix`. READ-ONLY analysis; no GPU booted; the background B=4 sweep was not
disturbed. Every number below is read from a named artifact / source line; the only arithmetic
performed is dividing the two counters the producing script itself divides (formula verified in
source first — per `feedback_dont_handroll_speed_defer_tuning` /
`feedback_check_artifact_before_concluding`).

---

## TL;DR

The user's historical "~40 aggregate" is **39.9065 decode_tps** from the REAL 16-task SWE-bench-Verified
deployment campaign `fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z` (native MTP-5, B=4). Our current
`aggregate_decode_tps = 15.66` is the **SAME accounting family** — `generation_tokens / idle-inclusive
agentic wall` — NOT a different denominator and NOT an idle-handling asymmetry (the prime-suspect lead is
**REFUTED**: both walls include agentic idle).

The 2.548x gap factors EXACTLY into two physical effects:

```
aggregate = per_stream_decode_rate  x  effective_concurrency
HISTORY:    39.9065 = 14.757 x 2.704
OURS:       15.6639 =  7.562 x 2.071
GAP 2.548x =          1.951x x 1.306x
```

- **1.951x = a real PER-STREAM slowdown** (history 14.757 vs ours 7.562 gen/decode-second). This is the
  genuine speed question, partly workload (4 astropy tasks vs a 16-task mix), possibly partly a real
  per-forward cost delta — chase it on a matched config, do NOT hand-wave it away.
- **1.306x = lower effective concurrency** (history sustained eff-conc 2.704 over 8718s of 16 tasks; ours
  2.071 over a ramp/drain-dominated 3844s of only 4 tasks).

Do NOT compare history 39.9 against our `derived_tps 7.50` or `per_request_decode_tps 6.53` — those are
per-request / concurrency-summed bases (different denominators). The only history-comparable field is
`aggregate_decode_tps`.

---

## 1. The ~40 pinned

| field | value | regime | source |
|---|---|---|---|
| **decode_tps** | **39.90650600912407** | REAL SWE-Verified deployment (codex agent loop, n=16, B=4, native MTP-5, fp8, temp 0.6, 8/16 resolved) | `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/agentic_summary.json` → `steptrace.decode_tps` |

Exact accounting (`scripts/summarize_round_f_agentic_arm.py:99-123`, `_step_summary`):

```
dt  = rows[-1].ts - rows[0].ts          # L102, ts span of step_rows FILTERED to driver-start -> last-task-end (L194-205)
gen = delta("gen")                      # L103, vllm:generation_tokens_total delta, AGGREGATE (summed over the B=4 streams)
decode_tps = gen / dt                   # L113
```

Verified arithmetic: `347914 / 8718.227547 = 39.90650600912407` (exact).

**`window_s` (dt) = the FULL campaign agentic wall, idle-INCLUDED** — proven three ways:
- `campaign_summary.json`: `ended_at − started_at = 03:24:28Z − 00:59:17Z = 8711s` ≈ `window_s 8718s` (~7s monitor lead).
- `codex_wall_seconds.p50 = 1800.1s` (full 30-min agent walls), `eval_wall_seconds.p50 = 51.9s` (x86 eval) are
  inside that window → tool calls + inter-turn idle + eval are all in the denominator.
- `engine_steps × mean_engine_step_ms_wall = 24563 × 354.9333 / 1000 = 8718.23` is a TAUTOLOGY
  (`mean_engine_step_ms_wall = dt*1000/iters`, L123), NOT independent evidence of an "engine-busy" wall.
  (The raw `dgx_steptrace.jsonl` ts span is 105,600s — that is pre/post-campaign monitoring, filtered out
  by L194-205; do not confuse it with `window_s`.)

regime CONFIRMED real deployment: `driver.log:1` = `dataset=princeton-nlp/SWE-bench_Verified ... n=16
concurrency=4`; per-task dirs have `codex_trace.jsonl`, `eval/`, `predictions.jsonl`. Propagated as the
accepted E5 baseline (`baseline_decode_tps 39.90650600912407`) in `fr9_*_s2_*/speed_comparison.json` and
`docs/reports/auto_research/fr9-b4-temp06-options-closeout-20260601.md:384`; named "39.9 decode TPS" across
`FR10_STATUS.md` (L117/478/532).

---

## 2. Reconciliation — WHY 39.9 vs 15.66

**It is regime (a) — SAME family, SAME idle-inclusive wall — driven by token-scale + concurrency + a real
per-stream gap.** It is NOT (b) per-request-vs-aggregate (we compare aggregate-to-aggregate), NOT a wall-
basis difference, and NOT (d) synthetic-mislabeled (the 39.9 run is unambiguously real deployment).

Our `aggregate_decode_tps` (`scripts/fr13_measure.py:1414-1429`) is the identical formula on our run:

```
earliest_pre.mtime -> latest_post.mtime   # UNION agentic wall across the n=4 task brackets (idle INCLUDED)
aggregate_decode_tps = (gen1 - gen0) / wall
                     = (60230 - 16) / 3844.1313 = 60214 / 3844.13 = 15.6639   # verified on-disk
```

Both denominators are idle-inclusive agentic walls. The gap decomposes exactly because
`aggregate = per_stream_rate × effective_concurrency` where `effective_concurrency = decode_seconds_sum /
wall`:

| | per-stream gen/dec_s | eff-conc dec_s/wall | aggregate gen/wall |
|---|---|---|---|
| HISTORY (n=16) | 347914/23575.92 = **14.757** | 23575.92/8718.23 = **2.704** | **39.907** |
| OURS (n=4) | 60214/7962.49 = **7.562** | 7962.49/3844.13 = **2.071** | **15.664** |
| ratio | **1.951x** | **1.306x** | **2.548x** |

`1.951 × 1.306 = 2.548 = 39.907 / 15.664` (closes exactly).

Sanity (independent of the union basis): the synthetic `fr12_deliverable_swe4_probe` (run #2 below) has its
OWN `warm_decode_tps = gen/decode_seconds = 2048/127.75 = 16.03` ≈ our 15.66, while its `returned_tokens /
wall = 41.27`. Same kernel, ~16 vs ~41 purely from denominator family — confirming ~16-class and ~40-class
are basis siblings, not a regression. But that probe is NOT the deployment number; the user's ~40 is run #1.

---

## 3. Same regime or synthetic — definitively

| # | number | value | regime | artifact |
|---|---|---|---|---|
| 1 | **decode_tps** | **39.9065** | **REAL SWE-Verified deployment, n=16** ← THE USER'S ~40 | `fr9_..._004903Z/agentic_summary.json` steptrace |
| 2 | returned_tokens_per_wall_s | 41.266 | SYNTHETIC `/v1/completions` probe (16×128-tok burst, NO codex loop) | `fr13_swe_verified_b4_diag_20260609T190931Z/native_b4_swe4/native_b4_swe4_probe.json` |
| 3 | warm_decode_tps (probe #2) | 16.031 | SYNTHETIC (= our 15.66 sibling) | same probe.json |
| 4 | quick_native_mtp5_b4 | 45.06 agg / 47.3 per-req / 15.65 warm | SYNTHETIC toy-prompt probe (max_tokens 64) | `fr13_argmax_e2e_20260608T055851Z/native_mtp5/quick_native_mtp5_b4.json` |
| 5 | fr10 starting point naive_mtp | 26.2 agg / 8.04 warm | SYNTHETIC toy probe (BATCH_INVARIANT-on "8 TPS" artifact) | `output/fr10_speed_starting_point/quick_decode_tps_tree_vs_naive.json` |

`41.266` (`scripts/fr12_deliverable_swe4_probe.py:260`, `wall_s = time.time()-t0`) is superficially ~40 but
is a synthetic probe — its own bind `FR13_SWE_B4_DIAGNOSTIC_BIND.md:18` states "It does not run the full
Codex SWE agent/evaluator loop." It only borrows 4 SWE prompts. The `FR13_LADDER_LOG.md:1161` phrasing
"Deployed-regime speed verdict ... 41.266" is a LOOSE LABEL — the artifact is a probe, not deployment.

**Verdict: the user's ~40 is RUN #1 (39.9065), genuine B=4 SWE-Verified deployment.** 41.266 is the wrong
regime and must be excluded for the deployment comparison.

---

## 4. Accounting map (apples-to-apples grouping)

### Group A — AGGREGATE gen/wall (the ONLY history↔ours comparable family)
| number | basis | numerator | denominator | source |
|---|---|---|---|---|
| **39.9065** (history) | gen / idle-inclusive campaign wall | gen 347914 (summed-streams) | window_s 8718.23 (idle in) | summarize_round_f_agentic_arm.py:113 |
| **15.6639** (ours) | gen / idle-inclusive union wall | gen 60214 (summed-streams) | union 3844.13 (idle in) | fr13_measure.py:1417-1429 |
| 41.266 (synth #2) | returned / client wall | 2048 | 49.629 (no agent idle) | fr12_deliverable_swe4_probe.py:260 |
| 45.06 (synth #4) | returned / client wall | toy | toy | quick_native_mtp5_b4.json |

### Group B — PER-STREAM decode rate (NOT directly history-comparable; history has no TPOT)
| number | basis | source |
|---|---|---|
| 14.757 (history derived) | gen / request_decode_time_seconds_sum | computed from steptrace counters |
| 7.562 (ours derived) | gen_union / dec_sum_union | computed from our brackets |
| **6.534** (ours per_request_decode_tps) | count/sum of request_time_per_output_token = 1/avg TPOT | fr13_measure.py:1411 |
| 11.875 (synth #2 request_tps_mean) | mean(tok/request_elapsed) | fr12_deliverable_swe4_probe.py:244-248 |

### Group C — CONCURRENCY-SUMMED (FR10-flagged; NEVER E5-comparable at B>1)
| number | basis | source |
|---|---|---|
| **7.5006** (ours derived_tps) | gen / decode_sum_s, decode_sum summed over co-resident streams ⇒ no concurrency credit (also double-counts via the summed agg: agg gen 178454) | fr13_measure.py:1405 |
| 8.04 (synth #5 warm) | gen / decode_sum (+ BATCH_INVARIANT artifact) | quick_decode_tps_tree_vs_naive.json |

### Group D — ct/ds (the SWE harness "official E5 comparison" metric)
`scripts/full_data_sweep.py:72-85`: per-call `decode_tps = completion_tokens / decode_sum_s`, then
`safe_median` across calls. `decode_sum_s ← vllm:request_decode_time_seconds_sum`
(`inference_proxy.py:826`). It is a **per-request decode-active rate, MEDIAN-aggregated** — i.e. a Group-B
sibling, NOT a wall and NOT concurrency-summed. **~40 is NOT the ct/ds median**; ct/ds is in the ~6-15
per-stream band. The headline 39.9 is the steptrace `gen/wall` aggregate, a different family from ct/ds.

---

## 5. Recommended apples-to-apple metric

**Compare ONLY `aggregate_decode_tps` (gen / idle-inclusive wall) against history 39.9, on a task-count- and
concurrency-matched campaign (ideally 16 SWE-Verified tasks at B=4 like E5).** That is the same family. Do
NOT compare 39.9 to `derived_tps 7.50` or `per_request_decode_tps 6.53`.

Two caveats before declaring parity:
1. The 1.951x **per-stream** gap is real and is the substantive speed question — it persists after removing
   the concurrency factor and is NOT explained by idle or basis. Diagnose it on a matched config (same task
   subset / seq-length mix); part is the 4-astropy-task workload (s/fwd 0.596), part may be a real
   per-forward delta.
2. accept/event differs (history 3.027 vs ours 3.471) → different token mix; not strictly apples-to-apple
   on trajectory either.

**fr13_measure.py deploy-speed field to ADD** (so future reporting reproduces the ~40 basis directly):

> Add **`effective_concurrency`** = `d(request_decode_time_seconds_sum) / aggregate_window_wall_s` to the
> `aggregate` block (one line; both counters already scraped). Then publish the decomposition
> `aggregate_decode_tps = per_stream_rate × effective_concurrency` alongside the existing
> `aggregate_decode_tps`. With `effective_concurrency` present, any future run's 15.66-class number is
> instantly placed on the 39.9 ladder (multiply out / divide back), and the per-stream gap is separated
> from the concurrency gap automatically. Optionally also emit the steptrace `gen/window` form by enabling
> steptrace capture (our nativeE5_b4 brackets have no steptrace / `vllm_request_metrics.jsonl` is empty
> with `deferred_full_normalization=true`, so only the mtime-union wall is available today).

The existing `aggregate_decode_tps` IS the right field; it just needs the `effective_concurrency` companion
so it is never again mistaken for "slower than 40."

---

## 6. Sources (grounding)

- THE 39.9: `output/fr9_b4temp06_lowmem088_mtp5_s1_20260602T004903Z/agentic_summary.json`
  (`steptrace.decode_tps 39.90650600912407`, `generation_tokens 347914`, `window_s 8718.227547`,
  `request_decode_time_s 23575.919`, `accept_per_event_steptrace 3.0265`, `engine_steps 24563`); `driver.log:1`;
  `campaign_summary.json` (`started 00:59:17Z`/`ended 03:24:28Z`, `resolved 8/16`,
  `codex_wall_seconds.p50 1800.1`, `eval_wall_seconds.p50 51.9`).
- OURS 15.66/7.50/6.53: `output/fr13_bigdenom_swe/nativeE5_b4/deploy_speed_b4.json`
  (`aggregate_decode_tps 15.663877`, `aggregate_window_wall_s 3844.131`, `derived_tps 7.500554`,
  `per_request_decode_tps 6.534427`, `raw_counter_delta_aggregate.generation_tokens_total 178454`,
  `request_decode_time_seconds_sum 23789.97`); per-task `vllm_metrics_pre/post.txt`
  (gen 16→60230, dec_sum 1.177→7963.67 over the union 3844.13s).
- SYNTHETIC 41.266: `output/fr13_swe_verified_b4_diag_20260609T190931Z/native_b4_swe4/native_b4_swe4_probe.json`
  (`returned_tokens_per_wall_s 41.266`, `wall_s 49.629`, `warm_decode_tps 16.031`); bind
  `FR13_SWE_B4_DIAGNOSTIC_BIND.md:18,44`; `FR13_LADDER_LOG.md:1158-1161` (commit df7a86e9).
- Source lines: `scripts/summarize_round_f_agentic_arm.py:99-123` (steptrace gen/dt);
  `scripts/fr12_deliverable_swe4_probe.py:242-272` (returned/wall);
  `scripts/fr13_measure.py:1405,1411,1414-1429` (derived/per_request/aggregate);
  `scripts/full_data_sweep.py:72-85` (ct/ds per-request median); `src/lumo_flywheel_serving/inference_proxy.py:826`
  (`decode_sum_s ← vllm:request_decode_time_seconds_sum`).
- Prior partial: `FR13_SPEED_HISTORY_RECONCILE.md` (B=1 per-forward tax only; does NOT cover this deployment
  aggregate) — this doc supersedes it for the B=4 aggregate question.
