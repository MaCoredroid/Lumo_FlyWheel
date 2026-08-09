# B4 promotion criterion — aggregate TPS is not sufficient (2026-08-09)

## The defect

Until now a B4 candidate arm was judged on one number, recorded in the timing
summaries as `decision_metric: measured_tps_fullstep_wall`. That number is an
**aggregate**: committed tokens per second summed over every co-resident
request. It is the product of two independent things:

```
measured_tps_fullstep_wall  =  per_request_step_tps  x  events_per_step
                               ^^^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
                               how fast one request     how many requests
                               is actually served       happened to be resident
```

Only the first factor is a property of the kernel stack. The second is a
property of how the campaign's four agent sessions happened to overlap on that
particular run — task-end skew, agent think time, container setup. A candidate
can therefore post a large aggregate gain **without making anything faster**,
purely by being measured on a run where more requests were resident.

## The measured case that forced this

The two-M candidate cleared the aggregate metric decisively and would have been
promoted on it:

| | stock | two-M candidate | delta |
|---|---|---|---|
| `measured_tps_fullstep_wall` (aggregate) | — | — | **+17.2%** |
| `per_request_step_tps` (batch-invariant) | — | — | **−2.96%** |

The implied co-residency ratio is `1.172 / 0.9704 = 1.2077`, i.e. the candidate
arm was measured with ~21% more requests resident per step. The entire aggregate
gain is that ratio. Every individual agent session was **slower** under the
candidate than under stock.

Promoting it would have shipped a regression and then baked the scheduling luck
into the baseline, so the next candidate would have had to beat a number that
was never a real speed.

## The criterion

A candidate is `promotion_eligible` iff **both** hold:

1. **Non-regression on the batch-invariant rate.**
   `candidate.per_request_step_tps >= stock.per_request_step_tps`.
   This is the condition that cannot be bought with co-residency.

2. **A real aggregate gain.**
   `candidate.measured_tps_fullstep_wall > stock.measured_tps_fullstep_wall`.

Condition 2 alone is the old rule. Condition 1 is the new one, and it is what
rejects the two-M shape. Condition 2 is retained because a candidate that is
per-request neutral and aggregate neutral is not worth the risk of a change.

Note the asymmetry: condition 1 is `>=` and condition 2 is `>`. A candidate is
allowed to leave per-request throughput exactly unchanged (a pure co-residency
or scheduling improvement is still real work), but it must move the aggregate
somewhere to be worth promoting.

## Fields

`scripts/fr13_b4_timing_math.py::phase_breakdown` now reports both rates per
arm:

- `measured_tps_fullstep_wall` — aggregate, carried through from the record.
- `per_request_step_tps` — `committed_per_event / (step_wall_ms / 1000)`.
  Reconciled against `measured_tps_fullstep_wall / events_per_step`; the
  breakdown fails loud if the two disagree.

`scripts/fr13_b4_timing_math.py::promotion_verdict(stock, candidate)` returns
the verdict block, emitted as `promotion` in the timing summaries written by
`scripts/fr13_run_b4_draft_head_m32_timing.sh` and
`scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh`:

| field | meaning |
|---|---|
| `promotion_eligible` | both conditions hold |
| `per_request_non_regression` | condition 1 |
| `aggregate_gain` | condition 2 |
| `stock_per_request_step_tps` / `candidate_per_request_step_tps` | the batch-invariant rates |
| `per_request_step_tps_delta_frac` | signed fraction |
| `stock_measured_tps_fullstep_wall` / `candidate_measured_tps_fullstep_wall` | the aggregates |
| `measured_tps_fullstep_wall_delta_frac` | signed fraction |
| `reason` | why, in words — names the co-residency case explicitly |

## Scope and limits

This criterion decides **promotion**, not **acceptance**. Acceptance against the
hardware floor is still the one-sided U95 of full-step wall latency versus the
137.607 ms cap, and is unaffected.

`per_request_step_tps` is batch-invariant but it is not co-residency-corrected:
it does not tell you what the candidate *would* have done at the stock arm's
co-residency, because serving more requests concurrently genuinely costs
per-request latency. It is a non-regression guard, not a counterfactual. The
honest way to remove the confound is to compare arms at matched co-residency,
which is what the task-pool refill work (`FR13_B4_TASK_REFILL`) exists to make
possible — a pool larger than the slot count holds the served width steady
instead of letting it decay 4 -> 3 -> 2 -> 1 as sessions end.

Until arms are matched on co-residency, condition 1 is what stops scheduling
luck from being recorded as speed.
