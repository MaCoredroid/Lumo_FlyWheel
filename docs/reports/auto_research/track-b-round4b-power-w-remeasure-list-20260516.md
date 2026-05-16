# Round 4b — power_w-based remeasure list (all 5 ablation points)

Generated: 2026-05-16
Supersedes the narrow D-point-only contamination report (`track-b-round4b-ablation-d-point-contamination-20260514.md`) by sweeping ALL ablation points (D, A, B, C, OFF) for `power_w` anomalies and cross-referencing with `decode_tps`.

## TL;DR

Sweep across 11 tasks × 5 ablation points × 4 attempts (~220 cells worth of dcgm + vllm_request_metrics data) finds **only 4 attempts with confirmed contamination signature** (high `power_w` AND low `decode_tps` simultaneously). The other "power anomalies" on A/C are *not* contamination — they reflect that A and C points naturally run the GPU hotter than D (spec-decode toggles T2/T3 do more work per token), and tps stays at the cell-max in those attempts.

**Required remeasures: 4 attempts (P1, contaminated).**
**Recommended remeasures: 5 D-point attempts (P3, tps-low without power signal — could be model behavior, lower confidence).**
**A/C "power anomalies" (P2, 13 attempts): NOT remeasures — natural envelope of those ablation points.**

## Methodology

For every per-attempt `dcgm_samples.jsonl` + `vllm_request_metrics.jsonl` pair:

- `decode_tps_med` = median of per-call `completion_tokens / decode_sum_s` (final-output speed, draft tokens excluded)
- `power_w_med` = median power_w over the attempt's sampling window
- Cell reference is the **cleanest attempt** (cell_tps_max for throughput, cell_pwr_min for power) — contamination only pushes power UP and tps DOWN, so cell median is biased by contamination itself and cannot be used.

Three flag tiers:

| Tier | Rule | Interpretation |
|---|---|---|
| **P1 CONTAM** | `tps < 0.55 × cell-max` AND `pwr > 1.15 × cell-min` | Host contention confirmed — work-power inversion |
| **P2 PWR-HI** | `pwr > 42W` AND `pwr > 1.10 × cell-min`, tps may be normal | High power, but if tps is also high it's just "GPU working hard" not contamination |
| **P3 TPS-LO** | `tps < 0.50 × cell-max`, no power signal | Throughput dip with normal power — likely model behavior (long thinking turn, short completion) |

## Priority 1 — CONFIRMED CONTAMINATION (must remeasure)

| Point | Task | Attempt | tps | tps/max | pwr_W | pwr/min |
|---|---|---|---:|---:|---:|---:|
| **D** | responses-sdk-adapter-cutover | run_02 | 5.10 | 0.18× | 45.2 | 1.25× |
| **D** | responses-sdk-adapter-cutover | run_03 | 7.85 | 0.27× | 46.6 | 1.29× |
| **D** | transcript-merge-regression | run_02 | 9.33 | 0.36× | 43.2 | 1.16× |
| **D** | transcript-merge-regression | run_03 | 4.84 | 0.19× | 43.5 | 1.17× |

These four attempts cluster temporally (within ~2 hours on 2026-05-12 evening) and show the canonical contamination signature: power spike + throughput crash simultaneously. Median power 7-10W above cell-min while throughput drops to 0.18-0.36× of the cell-max attempt.

Cleaned D medians after dropping these 4 attempts:

- responses-sdk D: median(16.45, 29.11) ≈ **22.78 tps** (vs raw cell median ~11)
- transcript-merge D: median(25.91, 11.81) ≈ **18.86 tps** (vs raw cell median ~11)

These two re-medians fully revise the prior conclusion in `track-b-round4b-ablation-d-point-contamination-20260514.md`.

## Priority 2 — POWER ANOMALY (NOT remeasures — natural envelope of A/C)

These 13 attempts have `pwr > 42W` and `> 1.10× cell-min`, but **`tps is at or near cell-max`** for many of them. The work-power relationship is preserved (more tokens → more power), so these aren't contamination — they're the natural higher power envelope of A-point and C-point compared to D-point.

| Point | Task | Attempt | tps | tps/max | pwr_W |
|---|---|---|---:|---:|---:|
| A | dead-flag-reachability-audit | run_04 | 28.95 | 1.00× | 46.1 |
| A | fanout-fullstack-release-blocker | run_04 | 19.31 | 0.97× | 44.8 |
| A | incident-evidence-synthesis | run_01 | 22.07 | 0.62× | 45.8 |
| A | incident-evidence-synthesis | run_02 | 18.51 | 0.52× | 45.3 |
| A | incident-evidence-synthesis | run_04 | 35.52 | 1.00× | 44.5 |
| A | responses-sdk-adapter-cutover | run_03 | 34.64 | 1.00× | 44.8 |
| A | responses-sdk-adapter-cutover | run_04 | 30.19 | 0.87× | 46.1 |
| C | incident-evidence-synthesis | run_02 | 30.43 | 1.00× | 45.5 |
| C | incident-evidence-synthesis | run_03 | 22.72 | 0.75× | 45.0 |
| C | policy-aware-request-resolution | run_02 | 25.56 | 1.00× | 45.5 |
| C | policy-aware-request-resolution | run_04 | 24.54 | 0.96× | 46.4 |
| C | responses-sdk-adapter-cutover | run_01 | 27.97 | 0.97× | 44.8 |
| C | responses-sdk-adapter-cutover | run_03 | 28.78 | 1.00× | 45.7 |

**Why these aren't contamination:** OFF-point sits at 37.5W, D-point at 35-40W, A-point at 40-46W, C-point at 39-47W. As speculative-decode techniques are layered in (D→A adds T1 only; C adds T2+T3+T4), the GPU draws more power per unit time because more parallel compute is happening (multiple draft heads, verifier pass). Higher power **with** higher tps is the expected signature of more aggressive spec-decode. Don't remeasure.

## Priority 3 — TPS-LOW only (optional, lower confidence)

These 12 attempts have throughput dips with **no power signal**. Could be model behavior (a turn with one long completion that doesn't fit the per-call tps metric well) or a real measurement issue we can't distinguish from data alone.

| Point | Task | Attempt | tps | tps/max | pwr_W |
|---|---|---|---:|---:|---:|
| A | dead-flag-reachability-audit | run_01 | 13.74 | 0.47× | 42.6 |
| A | dead-flag-reachability-audit | run_03 | 13.80 | 0.48× | 41.9 |
| A | responses-sdk-adapter-cutover | run_01 | 16.76 | 0.48× | 42.1 |
| A | responses-sdk-adapter-cutover | run_02 | 12.27 | 0.35× | 40.4 |
| C | sqlalchemy-2-session-modernization | run_04 | 15.88 | 0.50× | 43.2 |
| D | dead-flag-reachability-audit | run_02 | 14.79 | 0.49× | 37.2 |
| D | incident-evidence-synthesis | run_01 | 13.87 | 0.44× | 35.0 |
| D | multi-tool-transaction-repair | run_01 | 12.18 | 0.47× | 33.6 |
| D | multi-tool-transaction-repair | run_03 | 10.09 | 0.39× | 34.1 |
| D | transcript-merge-regression | run_04 | 11.81 | 0.46× | 37.3 |
| OFF | multi-tool-transaction-repair | run_04 | 2.18 | 0.42× | 37.2 |
| OFF | responsive-checkout-visual-regression | run_01 | 2.62 | 0.40× | 37.5 |

**Recommendation:** Skip these unless final aggregate medians look unstable. The OFF entries especially are noise — OFF tps is 2-6 tps absolute, so a 0.42× ratio is 2-3 tps gap, not a measurement crisis.

## Aggregate impact on Round 4b conclusions

| Statistic | Before cleanup | After P1-only cleanup |
|---|---:|---:|
| D-point median tps across 11 tasks | ~17.0 (contaminated) | ~19.7 |
| A vs D net delta | +15% A wins | +10% A wins |
| D-wins-on (sqlalchemy, dead-flag, multi-tool) | unchanged | unchanged |
| "+101% A wins on responses-sdk" | spurious | flat (−5%) |
| "+34% A wins on transcript-merge" | spurious | D wins (−17%) |

The aggregate "A is roughly flat to slightly favored over D" conclusion survives. The per-task attribution for responses-sdk and transcript-merge flips after cleanup, but those were always the two suspicious cells.

## Why my prior sweeps missed half the contamination

`track-b-round4b-ablation-d-point-contamination-20260514.md` correctly identified the 4 P1 attempts via D-point-only inspection. The first variance-gate I tried (using cell median as reference) flagged only responses-sdk D and entirely missed transcript-merge D — because the contaminated attempts themselves dragged the cell median down enough that "ratio to median" looked benign. **The cell MAX/MIN reference is the right denominator for small-N cells**: max-tps and min-power approximate the cleanest available reference point inside the cell.

Add this lesson to the round driver's variance gate (item 3 in the prior report).

## Concrete remeasure plan

**Required (P1) — 4 attempts × 1 task each:**

1. Re-run `responses-sdk-adapter-cutover` D-point, attempts run_02 and run_03 — 2 × 30 min = 1 hr wall
2. Re-run `transcript-merge-regression` D-point, attempts run_02 and run_03 — 2 × 30 min = 1 hr wall

**Total: ~2 wall-hours, schedule during a known-quiet host window.**

Acceptance: new attempts should show `decode_tps` in 22-30 tps band and `power_w` in 36-40W band (matching clean run_01/run_04). If they instead match the contaminated profile (5-9 tps, 43-47W), the contamination is recurring and we need to investigate the host (other tenants on the box, cron jobs, dcgm-exporter throughput).

**Skip (P2):** Do not remeasure A or C power-anomaly attempts. They reflect the natural higher-power envelope of A-only (T1) and C (T1+T2+T3) compared to D (all on).

**Optional (P3):** Skip unless final medians wobble. The 5 D-point P3 attempts are spread across 4 different tasks and look more like model-behavior variance than measurement artifacts.

## Reproduce

```bash
python3 -c '
import json, statistics
from pathlib import Path
ROOT = Path("output")
POINTS = {
    "D": [ROOT/"track_b_e2e_v4a_v2"/"round_0",
          ROOT/"track_b_e2e_v4a_v2"/"round_0_phase1_task1_2_PRESERVED",
          ROOT/"track_b_e2e_v4a_v2"/"round_0_phase2_task3_4_PRESERVED",
          ROOT/"track_b_e2e_v4a_v2"/"round_0_phase3a_PRESERVED"],
    "A": [ROOT/"track_b_e2e_v4a_v2_ablation"/"round_1"],
    "B": [ROOT/"track_b_e2e_v4a_v2_ablation"/"round_2"],
    "C": [ROOT/"track_b_e2e_v4a_v2_ablation"/"round_3"],
    "OFF": [ROOT/"track_b_e2e_v4a_v2_ablation"/"round_4"],
}
for pt, roots in POINTS.items():
    for parent in roots:
        if not parent.is_dir(): continue
        for task in sorted(parent.iterdir()):
            if not task.is_dir(): continue
            for run in sorted(task.iterdir()):
                if not run.is_dir() or not run.name.startswith("run_"): continue
                mfile = run/"vllm_request_metrics.jsonl"
                dfile = run/"dcgm_samples.jsonl"
                if not mfile.is_file(): continue
                tps=[(r["completion_tokens"]/r["decode_sum_s"]) for r in (json.loads(l) for l in mfile.read_text().splitlines() if l) if r.get("decode_sum_s",0)>0 and r.get("completion_tokens",0)>0]
                pwr=[d["power_w"] for d in (json.loads(l) for l in dfile.read_text().splitlines() if l) if d.get("power_w",0)>0] if dfile.is_file() else []
                if not tps: continue
                print(f"{pt} {task.name[:35]:35} {run.name} tps={statistics.median(tps):6.2f} pwr={statistics.median(pwr):5.1f}" if pwr else f"{pt} {task.name[:35]:35} {run.name} tps={statistics.median(tps):6.2f} pwr=n/a")
'
```

## Files

- This report: `docs/reports/auto_research/track-b-round4b-power-w-remeasure-list-20260516.md`
- Supersedes: `docs/reports/auto_research/track-b-round4b-ablation-d-point-contamination-20260514.md` (4-attempt list still correct, but methodology now generalized)
- Sweep script: `scripts/contamination_sweep.py` (TBD: stage from `/tmp/contamination_sweep_v2.py`)
- D-point data: `output/track_b_e2e_v4a_v2/round_0/`, `round_0_phase{1,2,3a}_PRESERVED/`
- A/B/C/OFF data: `output/track_b_e2e_v4a_v2_ablation/round_{1,2,3,4}/`
