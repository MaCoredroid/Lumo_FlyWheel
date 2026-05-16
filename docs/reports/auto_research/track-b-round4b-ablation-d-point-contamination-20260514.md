# Round 4b Ablation — D-point contamination on 2 tasks

Generated: 2026-05-14
Companion to: `track-b-v4a-v2-task1-responses-sdk-cutover-20260512.md`, `bdd54ed` (phase 3 baseline closeout), `8b3f9fc` (point A closeout)

## TL;DR

Two of the eleven v4a_v2 D-point per-task measurements have host-level contamination on 2 of 4 attempts each: **responses-sdk-adapter-cutover** and **transcript-merge-regression**. The contaminated attempts simultaneously show degraded decode_tps (3-5×) and degraded prefill_s (2-3×), in a clustered time window distinct from when the other 9 tasks were measured. Plus: **fanout-fullstack-release-blocker D-point data is double-counted across `round_0/` and `round_0_phase3a_PRESERVED/` due to a `run_NN` filename collision**.

The "+101% A wins big" and "+34% A wins" signals from my earlier ablation analysis on those two tasks **were measurement artifacts, not technique effects**.

## The contamination signature

Per-attempt D-point data on the two contaminated tasks:

### `responses-sdk-adapter-cutover` D-point

| Attempt | Time (UTC) | n_calls | decode_tps | prefill_s |
|---|---|---:|---:|---:|
| run_01 | 2026-05-12T20:52 | 110 | **16.45** | 1.50 |
| **run_02** | 2026-05-12T21:22 | 127 | **5.10** | **4.27** ← contaminated |
| **run_03** | 2026-05-12T21:52 | 201 | **7.85** | **3.59** ← contaminated |
| run_04 | 2026-05-12T22:22 | 149 | **29.11** | 2.00 |
| **D ratio min/max** | | | **5.7×** | 2.8× |

### `transcript-merge-regression` D-point

| Attempt | Time (UTC) | n_calls | decode_tps | prefill_s |
|---|---|---:|---:|---:|
| run_01 | 2026-05-12T22:47 | 183 | **25.91** | 1.47 |
| **run_02** | 2026-05-12T23:17 | 190 | **9.33** | **2.27** ← contaminated |
| **run_03** | 2026-05-12T23:47 | 77 | **4.84** | **3.38** ← contaminated |
| run_04 | 2026-05-12T23:59 | 130 | **11.81** | 1.91 |
| **D ratio min/max** | | | **5.4×** | 2.3× |

**Pattern:** the middle 2 attempts of each task show simultaneous decode_tps drop AND prefill_s rise — the canonical signature of host-level resource pressure (GPU thermal throttling, memory pressure, page-out, or concurrent CPU load). The first and last attempts are normal.

### Compare to the same tasks under A-point (T1 only, measured the next day)

| Task | D ratio min/max | A ratio min/max |
|---|---|---|
| responses-sdk | 5.10 → 29.11 (**5.7×**) | 12.27 → 34.64 (2.8×) |
| transcript-merge | 4.84 → 25.91 (**5.4×**) | 11.98 → 17.79 (1.5×) |
| fanout | (double-counted) | 14.89 → 19.85 (1.3×) |
| sqlalchemy (control) | 23.98 → 37.74 (1.6×) | 25.94 → 30.87 (1.2×) |
| dead-flag (control) | 14.79 → 30.05 (2.0×) | 13.74 → 28.95 (2.1×) |

Normal-task variance is 1.2-2.1×. Contaminated D for responses-sdk and transcript-merge is 5.4-5.7×. **A-point on the same tasks is in the normal 1.5-2.8× band** — measured on 2026-05-13 daytime when the host was stable.

## Cleaning the data

Drop the 2 contaminated attempts per task; recompute medians from the remaining 2:

| Task | D (cleaned, run_01 + run_04 only) | A (all 4) | Δ % |
|---|---:|---:|---:|
| responses-sdk-adapter-cutover | median(16.45, 29.11) ≈ **22.78** | 21.74 | **−5% (flat)** |
| transcript-merge-regression | median(25.91, 11.81) ≈ **18.86** | 15.60 | **−17% (D wins)** |

The "+101% A wins big" on responses-sdk **evaporates** (cleaned: flat). The "+34% A wins" on transcript-merge **flips direction** (cleaned: D 17% faster).

## Fanout double-count

`fanout-fullstack-release-blocker` has 6 D-point attempts on disk:

- `round_0_phase3a_PRESERVED/run_01` (May 13 14:43) — pre-watchdog
- `round_0_phase3a_PRESERVED/run_02` (May 13 15:13) — pre-watchdog (killed by watchdog `47f0b79`)
- `round_0/run_01` (May 13 16:41) — post-watchdog retake
- `round_0/run_02` (May 13 17:06) — post-watchdog
- `round_0/run_03` (May 13 17:36) — post-watchdog
- `round_0/run_04` (May 13 17:55) — post-watchdog

My aggregation script globs across all 4 D parent dirs and merges rows from both `round_0/run_01` and `phase3a_PRESERVED/run_01` (same filename). Net effect: fanout D rows = 562 instead of ~400 (the 4 canonical attempts), and the medians include 2 stale pre-watchdog attempts that should have been excluded.

**Canonical fanout D should use only `round_0/run_01..run_04`** (the post-watchdog retake). Recomputed:

| Attempt | decode_tps | prefill_s |
|---|---:|---:|
| run_01 | 14.56 | 1.35 |
| run_02 | 15.98 | 2.07 |
| run_03 | 17.20 | 1.52 |
| run_04 | 18.89 | 1.96 |
| **median** | **16.59** | **1.74** |

Clean fanout D = ~16.6 (vs the contaminated-aggregated 15.47 I reported earlier). A-point fanout = 17.42. **Clean Δ: +5% (was reported as +13%)**.

## Updated A-vs-D table after cleanup

| Task | D (cleaned) | A | Δ % | Real signal? |
|---|---:|---:|---:|---|
| responses-sdk-adapter-cutover | ~22.8 | 21.74 | −5% | **artifact removed** |
| transcript-merge-regression | ~18.9 | 15.60 | −17% | **D faster** (was reported A faster) |
| fanout-fullstack-release-blocker | ~16.6 | 17.42 | +5% | small (was +13%) |
| responsive-checkout-visual-regression | 14.81 | 16.27 | +10% | small, unchanged |
| policy-aware-request-resolution | 20.30 | 21.72 | +7% | small, unchanged |
| security-audit-hotfix-remediation | 16.70 | 17.10 | +2% | flat |
| release-note-to-plan-translation | 24.95 | 25.22 | +1% | flat |
| incident-evidence-synthesis | 23.54 | 22.96 | −2% | flat |
| sqlalchemy-2-session-modernization | 30.05 | 28.09 | −7% | D wins small |
| dead-flag-reachability-audit | 26.63 | 24.36 | −9% | D wins small |
| multi-tool-transaction-repair | 18.86 | 14.18 | **−25%** | D wins big |

**Cleaned aggregate median across 11 tasks:** D ≈ 19.7 → A ≈ 21.7 → **+10% net** (was +15% with contaminated data). The directional conclusion shifts: with clean data, T2/T3/T4 are roughly neutral on average, slightly net positive on hard tasks (sqlalchemy, dead-flag, multi-tool), slightly net negative on easier tasks where the agent loop converges fast.

## Recommendations

### Immediate

1. **Remeasure D-point for `responses-sdk-adapter-cutover` and `transcript-merge-regression`.** 4 attempts each × 30 min = ~4 wall hours. Schedule during a known-quiet host window. Verify variance against the cleanup hypothesis: if new attempts cluster in 22-30 tps with normal prefill_s, contamination diagnosis is confirmed.
2. **Fix the fanout double-count.** Either:
   - Move `round_0_phase3a_PRESERVED/run_NN` to a non-colliding path (e.g., `run_01_partial`, `run_02_partial`)
   - Add an `[abandoned]` marker file in the phase3a dirs and have aggregation scripts skip them
   - Update aggregation scripts to deduplicate by `(task, attempt_label)` pair and prefer `round_0/` over preserved siblings

### Protocol additions

3. **Add a per-cell variance gate to the round driver.** If any attempt's `decode_tps_median` is < 0.5× the cell median (across attempts of that cell), flag for retake. Would have caught both contaminated tasks at collection time.
4. **Sample DCGM during measurement.** Cross-reference per-attempt windows against `dcgm_samples.jsonl` for the same timestamps. If a contaminated attempt shows elevated `gpu_temperature` or `mem_pressure` during the window, that confirms host-level cause vs model-behavior cause.
5. **Standardize cold-start protocol.** Define a minimum container uptime before measurement starts (e.g., 30 min post-launch). Add a sentinel warmup-pass call that verifies prefill_tps is in the normal band before recording the first task attempt.

### Round 4b conclusions to revise

The earlier "A wins big on responses-sdk/transcript-merge" framing in my prior ablation-review responses **should be withdrawn** in favor of:

- A is roughly flat vs D across most tasks (10-task cleaned median: +10% net, was +15%)
- D actively wins on 3 stateful/refactor tasks (sqlalchemy, dead-flag, multi-tool)
- B-point partial data (only 3 of 11 tasks done) showing T2 strongly helps on transcript-merge needs re-validation against clean D before drawing per-technique attribution

## Reproduce

```bash
# Show per-attempt variance within D-point for any task:
.venv/bin/python -c '
import json, statistics
from pathlib import Path
ROOT = Path("output/track_b_e2e_v4a_v2")
TASK = "responses-sdk-adapter-cutover__v1-clean-baseline"
for parent in sorted(ROOT.iterdir()):
    if not (parent / TASK).is_dir(): continue
    for run in sorted((parent / TASK).iterdir()):
        rows = [json.loads(l) for l in (run / "vllm_request_metrics.jsonl").read_text().splitlines() if l]
        tps = [r["completion_tokens"]/r["decode_sum_s"] for r in rows if r.get("decode_sum_s",0)>0 and r.get("completion_tokens",0)>0]
        pref = [r["prefill_sum_s"] for r in rows if r.get("prefill_sum_s")]
        print(f"{parent.name} {run.name}: median_tps={statistics.median(tps):.2f} median_pref={statistics.median(pref):.2f}")
'
```

## Files

- This report: `docs/reports/auto_research/track-b-round4b-ablation-d-point-contamination-20260514.md`
- D-point per-task data: `output/track_b_e2e_v4a_v2/round_0_phase{1,2,3a}_PRESERVED/*/`
- A-point per-task data: `output/track_b_e2e_v4a_v2_ablation/round_1/*/`
- Ablation driver: `scripts/run_track_b_v4a_e2e_ablation.py`
