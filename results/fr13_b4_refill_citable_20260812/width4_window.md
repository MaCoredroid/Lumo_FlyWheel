# FR13 B4 — the width-4 depth window: an operating point, not a blend

Addendum to `design.md` §9, which closed the pool16 campaign `NOT_EVALUATED_INSUFFICIENT_PASSES`
and named the reason: **a 16-task pool at 4 slots is two regimes, wall-blended**, so
arm-level `events_per_step` is a mixture weight rather than a rate. §9 recommended
bracketing the metrics to the window where depth == slots. This is that instrument,
built and run.

* tool — `scripts/fr13_b4_width4_window_reduce.py`
* tests — `tests/test_fr13_b4_width4_window.py` (26)
* artifact — `fr13_b4_width4_operating_point.json` (this directory; also written next
  to the campaign under `output/`, which is gitignored)
* run class — `b4_width4_operating_point`, schema `fr13.b4_width4_operating_point.v1`,
  classification `real_swe_verified_b4_width4_operating_point`
* input — all **8 served arms** of `output/fr13_b4_pool16_refill_gate_20260812T115625Z`,
  including passes 2 and 3, which the arm-level 3.2 depth floor excluded

This class is **not citable and never will be**. It is an INSTRUMENT: it exists so the
step-wall levers (B4 `gqa_pair` re-test, `single_launch` route, F-window, prefill
isolation) have a stable target to be judged against. `citable` and
`formal_floor_acceptance_eligible` are both `false` by construction.

---

## 1. THE WINDOW

Over the admission ledger's events, ordered as `fr13_floor_gate._reduce_refill_ledger`
orders them (admits before completes at an identical timestamp):

```
open  = the first event at which depth == slots
close = the first `complete` event carrying pending == 0
```

`close` is the instant the pool stops being a pool. Depth falls below slots and there is
nothing unstarted left to refill it, so it can never return. For a 16-task / 4-slot arm
that is the **13th completion**, and it lands 3 admissions after the last admit — the
window therefore contains the whole admission sequence, which the tool asserts.

**Only `depth` and `pending` participate.** Both are integers the runner wrote before any
timing was reduced. No measured rate enters the boundary, so the window cannot be slid
toward a nicer number. That is the entire argument for why this is not cherry-picking,
and it is worth stating plainly because the shape of the operation — "measure only part
of the arm" — is exactly the shape of a laundered basis.

### Why nothing is interpolated

Three artifacts line up on the same events:

| | |
|---|---|
| `swe_out/verified/fr13_task_refill_ledger.jsonl` | timestamps every admit/complete with the resulting `depth` and `pending` |
| `swe_out/verified/per_task/<id>/vllm_metrics_{pre,post}.txt` | **real Prometheus snapshots taken at those same task boundaries** |
| `logs/fr13_fixed32_work_census.jsonl` | one record per forward step, absolute `forward_step_index` |

Both window edges are ledger events that own a snapshot, so the windowed counter delta is
a genuine bracket — the same `max(post) − min(pre)` envelope `fr13_measure` already uses
for a staggered arm, closed earlier. There is no wall→step regression, no assumed step
rate, no mtime arithmetic.

The bracket's own `fr13_decode_forward_gpu_steps_total` then *indexes* the census range,
and the census must agree **exactly**:

```
counter Δsteps  == number of census records in [origin, origin+Δsteps)
counter Δdrafts == Σ batch_size over those records
```

Enforced with `!=` and no tolerance, exactly as `fr13_measure.cmd_deploy_speed` enforces
it for a whole arm. On all 8 arms it passed on the first read — e.g. tail23 pass 0: 5545
counter steps, 5545 census records, 20168 events on both sides.

### Validation of the estimator against the shipped one

Before windowing anything, the tool's parse + envelope was checked against the sealed
arm-level artifacts: all **18 counters** of `raw_counter_delta_aggregate` reproduce
exactly, and every derived field (`step_wall_ms`, `measured_tps_fullstep_wall`,
`events_per_step`, `accept_per_event`, `prefill_frac`, `floor_ratio`, the GPU component
split) reproduces to 1e-9. The windowed record is then pushed through
`fr13_b4_timing_math.phase_breakdown` — the same function the arm-level reducer uses — so
every unit identity, including `aggregate == events_per_step × per_request`, is *enforced*
rather than assumed.

---

## 2. WHY THE DRAIN EXCLUSION IS HONEST BY CONSTRUCTION

The tool does not assert that the drain is excluded. It **proves it per arm**.

`_window_occupancy` accumulates full-width wall inside and outside the derived window
separately and requires the outside sum to be **exactly `0.0`** — an exact comparison, no
tolerance, valid because the quantity is a sum of non-negative terms and is either empty
or it is not. All 8 arms returned `full_width_wall_s_outside_window == 0.0`.

The consequence is the load-bearing one:

> **The window contains 100% of the arm's full-width wall.** It is not a sample of the
> full-width phase, it is the whole of it. Nothing that was ever at depth 4 was left
> behind, and nothing that was not at depth 4 was let in.

Two independent confirmations that the window is the right object:

1. `window_wall_fraction_of_arm` reproduces the ledger's own `full_width_fraction` to
   **1e-7** on every arm (0.8185242 vs 0.8185241, 0.7130706 vs 0.7130704, …). The window
   is the same object the pool gate was already measuring — this class just puts the
   *timers* inside it instead of counting its wall.
2. `depth_at_slots_fraction_within_window` = **0.9999997–0.9999998** on every arm. The
   window is at full width for all but ~1 part in 4 million of its wall; the residue is
   the sub-millisecond complete→admit handoffs (86 µs apiece).

The ledger summary is *recomputed* from the raw events and required to agree before any
of this runs (the house pattern — the runner is not trusted on its own word).

### What the exclusion costs, stated rather than hidden

The drain is **18.1%–64.8% of arm wall** on this campaign. So:

> A windowed rate multiplied by an arm wall would overstate delivered work, on pass 3 by
> nearly 3×. This class is **not** whole-arm throughput and must never be quoted as the
> arm's delivered rate.

`does_not_claim` carries this first, and the arm-level blended values ride alongside every
windowed value in the artifact under `arm_level_blended` (`role: "context only -- the
mixture this class decomposes"`).

### The arm-level 3.2 depth floor is deliberately NOT inherited

`time_weighted_mean_depth >= 3.2` asks *"was this arm mostly a pool?"* — a question about
the **mixture weight**, which is precisely what this class removes. Applying it would
discard the perfectly good full-width phases of arms whose *drains* were long. Passes 2
and 3 are in this reduction for exactly that reason, and their windows are as clean as
passes 0 and 1 (pass 3 hydra has the *largest* window of all eight, 5860 steps).

The ledger's **structural** invariants are still gated — schema, `slots`, `task_count`,
`completed`, `aborted`, `peak_depth <= slots` — because those ask whether the arm was a
valid pool run at all.

### Admissibility

| | |
|---|---|
| `MIN_WINDOW_STEPS` | **1000**, pinned as a constant, not fitted to the data it judges |
| basis | at ~385 ms/step that is ~6.4 min of sustained full-width serving, leaving ~880 wall-chain-retained samples behind the step mean |
| observed range | **4152 – 5860** steps |
| did the floor do any work? | **No.** The shortest window clears it by 4.15×. Reported in the artifact as `floor_did_work: false` |

The floor is a guard for future short-window arms. On this campaign it filtered nothing,
and the artifact says so rather than implying it was load-bearing.

---

## 3. THE RESULT

### Per arm

| pass | topo | events/step | per-req TPS | aggregate | step ms | window steps | window wall % | drain % excluded |
|---|---|---|---|---|---|---|---|---|
| 0 | T | 3.637 | 15.67 | 56.98 | 394.8 | 5545 | 71.3 | 28.7 |
| 0 | H | 3.600 | 16.01 | 57.63 | 389.6 | 4776 | 81.9 | 18.1 |
| 1 | T | 3.618 | 16.18 | 58.54 | 389.8 | 5328 | 75.0 | 25.0 |
| 1 | H | 3.573 | 16.32 | 58.31 | 388.6 | 4773 | 78.4 | 21.6 |
| 2 | T | 3.579 | 16.55 | 59.24 | 382.0 | 4486 | 53.0 | 47.0 |
| 2 | H | 3.611 | 15.76 | 56.90 | 385.4 | 5610 | 69.8 | 30.2 |
| 3 | T | 3.567 | 15.98 | 56.99 | 383.7 | 4152 | 35.2 | 64.8 |
| 3 | H | 3.678 | 14.63 | 53.82 | 394.8 | 5860 | 38.9 | 61.1 |

### Pooled by topology — n=4 per topology, df=3, `T95_ONE_SIDED[3]` = 2.3534

| statistic | Hydra27 | Tail23 |
|---|---|---|
| **per_request_step_tps** (primary) | **15.679** [14.815, 16.543] CV 4.68% | **16.094** [15.656, 16.531] CV 2.31% |
| measured_tps_fullstep_wall | 56.664 [54.331, 58.997] CV 3.50% | 57.938 [56.602, 59.275] CV 1.96% |
| events_per_step | 3.616 [3.563, 3.668] CV **1.23%** | 3.600 [3.562, 3.639] CV **0.91%** |
| step_wall_ms | 389.610 [385.010, 394.210] CV **1.00%** | 387.588 [380.681, 394.495] CV **1.51%** |

### The headline: the mixture weight is gone

| statistic | windowed CV | arm-level CV (all 4 passes) | tightening |
|---|---|---|---|
| events_per_step, H / T | 1.23% / 0.91% | 21.76% / 23.30% | **17.6× / 25.6×** |
| aggregate, H / T | 3.50% / 1.96% | 18.78% / 22.45% | 5.4× / 11.4× |
| step_wall_ms, H / T | 1.00% / 1.51% | 8.28% / 11.45% | 8.3× / 7.6× |
| per_request_step_tps, H / T | 4.68% / 2.31% | 3.54% / 6.38% | 0.8× / 2.8× |

Arm-level `events_per_step` spanned **1.67–3.04** across the campaign. Windowed it spans
**3.567–3.678** — a total range of 3.1%. That is the whole thesis, measured: the 1.67–3.04
spread was never service variability, it was the drain-tail fraction moving 0.18 → 0.65,
and the window removes it.

Note the honest exception: **per-request TPS did not tighten** (it got slightly *worse* on
Hydra). That is expected and confirming rather than disappointing — `per_request_step_tps`
is already the batch-invariant statistic, which is why the exact4 gate made it primary.
Windowing does not stabilise it; windowing **corrects** it.

### The correction, and it is bad news

| | windowed (this class) | arm-level blend, 2 included passes (§9) | exact4 ON gate, sealed |
|---|---|---|---|
| per-request, H / T | **15.68 / 16.09** | 16.99 / 17.44 | 21.20 / 22.38 |
| vs exact4 | **−26.0% / −28.1%** | −19.9% / −22.1% | — |

The blended number **flattered** per-request service by 8–8.4%. Mechanism: in the drain the
batch is narrow, so each surviving request takes a larger share of every step and posts a
higher per-request rate. Blending that in made width-4 service look better than it is.

**At true width 4 the per-request regression against exact4 is ~26–28%, not ~20–22%.** The
`per_request_non_regression` reading stays FALSE and gets worse under a sharper
instrument. This is the number Mark's optimization loop has to move.

*(Both comparisons against exact4 are descriptive only — a phase measured against a
mixture, on a different task set and a different bracket estimator. The aggregate contrast
is worse still and is deliberately not headlined: comparing a windowed pool16 aggregate to
a whole-arm exact4 aggregate is close to meaningless.)*

### What the window is NOT

Inside the window the **engine batch** was width 4 on only **62.5%–70.0%** of steps, width 3
on 22.3%–33.2%, narrower on the rest. The window is a **pool-depth** window: it selects the
wall during which four tasks were admitted and unfinished, not the steps at which the
engine held four rows. The residual is agent-side — an admitted task is repeatedly between
requests while it runs tools locally.

A true `batch == 4` filter is **not derivable from this evidence**: the census resolves
batch width per step but carries **no per-step wall**, so no wall statistic can be
conditioned on it. This is the single largest remaining gap and it is what §4 asks for.

### Inherited caveats, unchanged

* `step_wall_ms` and `events_per_step` still do not share a basis. Windowed
  `retained_wall_fraction` is **0.877–0.896**, *lower* than the arm-level 0.906–0.932 —
  admission churn resets the wall chain more often, and admission is densest exactly here.
  Whether the first step after a reset is slower in wall terms remains unmeasurable by
  construction.
* `prefill_frac` (0.182–0.255 windowed) and `per_request_decode_tps` come from
  `request_*_seconds_sum` counters that advance only on request **completion**, so requests
  in flight at the window edges bias them. They are emitted; they carry no intervals. The
  step-keyed statistics have no such bias.

---

## 4. STATISTICAL VERDICT

**Per topology: n=4 independent passes, df=3 — the repo's pinned critical applies exactly,
and intervals are emitted.**

**Pooled across topologies: refused.** 8 windows are not 8 draws. The two arms inside a
pass share host conditions and one task-difficulty draw (§9), so the independent unit is
the pass: 4 per topology, 8 pooled → **df=7, which `T95_ONE_SIDED` does not pin**. The tool
emits `status: "no_pinned_critical"` with the point estimate (15.886) and CV (3.67%) and
**no bounds**. No df=7 critical was invented to buy a tighter-looking number, exactly as
§3 refused to invent a df=1 critical for a 2-pass campaign.

### Minimum detectable effect — the number the levers will be judged against

At n=4 with the pinned critical, MDE ≈ CV × t/√n = CV × 1.1767:

| statistic | Hydra27 MDE | Tail23 MDE |
|---|---|---|
| **step_wall_ms** | **1.18%** | **1.78%** |
| events_per_step | 1.45% | 1.07% |
| aggregate | 4.12% | 2.31% |
| per_request_step_tps | 5.51% | 2.72% |

This is the deliverable. **A step-wall lever that moves `step_wall_ms` by more than ~1.2–1.8%
is now separable at four passes** — against 8.3%/11.5% arm-level CV, where nothing below
~10% was separable at any affordable N. The instrument bought roughly an order of magnitude
of power on the axis the levers actually act on.

Per-request needs a larger move (2.7–5.5%) because its CV did not tighten. If a lever is
expected to land inside that band, the honest options are 16 passes (df=15 is pinned) or a
paired within-campaign design, **not** a pooled df=7 interval.

---

## 5. WHAT A FUTURE ARM-LEVEL PROMETHEUS BRACKET AT DEPTH BOUNDARIES WOULD ADD

This reduction is bounded by the fact that the only Prometheus snapshots that exist are
**per-task** ones, taken at task start and task end. That was enough here *only because the
window edges happen to coincide with task boundaries* — the open is the 4th admit, the
close is the 13th completion. That coincidence is a property of a 16/4 pool, not a
guarantee.

An arm-level bracket **emitted by the serving path at every depth transition** —
`fr13_fixed32_depth_bracket.jsonl`, one counter snapshot per `depth` change with the
resulting depth — would add four things this reduction cannot have:

1. **Per-depth operating points, not just depth==slots.** Depth 3 and depth 2 phases would
   get their own bracketed rates. The step-wall-vs-width curve is currently inferred from
   two whole-campaign points (exact4 ≈ 271–280 ms, width4 ≈ 388–390 ms); it would become a
   measured curve inside a single arm, at zero extra GPU cost.
2. **Per-task phase alignment.** Right now a task's contribution cannot be attributed to
   the phase it ran in — the per-task bracket spans the task, and the task spans phases.
   Depth-boundary snapshots would cut per-task work at phase edges, which is what turns
   "the pool is slower at width 4" into "*this* task's decode is slower at width 4".
3. **Removal of the hydration lag from the window edge.** The ledger admit and the engine's
   first step for that task are separated by repo hydration — 118 forward steps (~43 s) on
   tail23 pass 0. A depth-transition bracket would be emitted by the *engine*, so the
   window would track engine-side co-residency instead of worker-thread depth, and
   `does_not_claim` item 3 (the 62–70% batch-4 residual) would shrink from a caveat into a
   measured quantity.
4. **A wall basis for the batch-width filter.** If the depth bracket carried the wall-chain
   counters, `step_wall_ms` could finally be conditioned on batch width — the one thing the
   census cannot do today.

Cost: one JSONL append per depth transition — **~30 lines per arm**, against the 9385-line
work census already written. This is the cheapest instrumentation left on the board and it
is the natural next offline-enabling change.

Until it exists, the honest statement is the one this class makes: the window is a
**pool-depth** window with a published engine-batch residual.

---

## 6. THE nsys IMPLICATION

§9 already said it; this reduction makes it quantitative.

**Profile inside the window only.** A width-3 nsys attribution taken over a whole pool16 arm
would sample the drain in proportion to its wall — **18% to 65% of the arm on this
campaign, and 61–65% on the two passes with the long stragglers.** Kernel time attributed
from a pass-3 arm would be *majority drain*: an exact4-shaped 4→3→2→1 wave profiled and
labelled as the width-4 operating point.

Concretely, for the next GPU rung:

| | |
|---|---|
| **what** | the width-4 operating point of the fixed32 B4 stack — decode step composition at events/step ≈ 3.6 |
| **when** | strictly between the ledger's 4th admit and its 13th completion. The artifact publishes both `forward_step_index` bounds per arm (`census_first_forward_step`, `census_last_forward_step`) |
| **how to align** | the window is already expressed in **absolute forward-step indices**, the same coordinate the work census uses. Gate the profiler on the step counter, not on wall time — wall-time gating would drift with the hydration lag |
| **how much** | the shortest window is 4152 steps; a capture of a few hundred consecutive steps sits comfortably inside any of the eight |
| **what to avoid** | starting the capture at arm start. The first steps of an arm run at width 1–2 while the initial four tasks hydrate, and they are inside the window but unrepresentative of it |
| **cross-check** | the capture's step count and event count must satisfy the same exact identity the census gate enforces, or the profile is not of the window it claims |

The step-wall levers all act on the same object this window measures — `step_wall_ms` at
width 4, currently **387.6 / 389.6 ms** against a 126.514 ms hardware floor (`floor_ratio`
3.064 / 3.080 at the topology means, 3.020–3.121 per arm). The GPU component split is published per arm in
`phase_breakdown`, so an nsys capture inside the window is directly reconcilable against
`sfwd / dfwd / cfwd / other` before a single kernel is renamed.

---

## 7. IMPLEMENTATION NOTES

* **A sibling tool, not an edit to `RUN_CLASSES`.** `tests/test_fr13_b4_pool16_refill_timing.py`
  pins `set(RUN_CLASSES) == {exact4_formal_floor, pool16_refill_timing}`, and both classes
  are cited by sealed artifacts. A third entry would mutate a registry that sealed evidence
  depends on. The new class carries its own schema/classification tokens and asserts they
  collide with neither neighbour.
* **Reuse over reimplementation.** `fr13_measure._scrape_metrics_file` and the `M_*`/`COUNTERS`
  constants for parsing, `fr13_b4_timing_math.phase_breakdown` for the identities,
  `fr13_b4_floor_gate_reduce.cluster_interval` / `T95_ONE_SIDED` / `exact_json` for the
  statistics and strict JSON. `fr13_floor_gate._reduce_refill_ledger` is *ported* (12901-line
  module, pulls the serving stack) and its arithmetic is unchanged.
* **No tolerances anywhere.** Census gate: `!=`. Drain-exclusion proof: `!= 0.0`. Bracket
  monotonicity: `<`. The only `isclose` is in the ledger-summary recomputation, at 1e-9,
  matching the upstream gate.
* **Every exclusion is named.** `reduce_window_arm` never raises for an evidence defect; it
  records `exclusion_reason` so the count of surviving draws is auditable. A single
  defective arm drops the campaign to `NOT_EVALUATED_INSUFFICIENT_WINDOWS`.
* **Atomic emission** — tmp + `replace`, `sort_keys=True`, `allow_nan=False`.
* **Tests**: 26, all offline, no container. Fixtures are synthetic ledgers/censuses/brackets
  built so `per_request_step_tps == 12.5` in closed form, which makes a boundary bug show up
  as a wrong *number* rather than only a wrong shape. Covered: clean window; a 6-task pool
  that never reached width (no window, not a short one); a straggler arm whose window is real
  but 125 steps and inadmissible; close-is-the-first-`pending==0`; full-width wall after the
  close refused; census hole; one-event census perturbation refused; a drain-step
  perturbation correctly invisible; summary-vs-events disagreement; pool-not-larger-than-slots;
  the depth floor demonstrably not applied; df=3 interval vs df=7 refusal; class round-trip.
