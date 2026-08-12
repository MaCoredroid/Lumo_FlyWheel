# FR13 B4 — the pool16 refill timing class

Branch `codex/fr13-b4-refill-citable-20260812`, off `main` 1e0158bf2 (which already
contains the whole b4-twom / formal-gate stack: pass atomicity 04f341633, campaign-commit
stack resolution af2325482, `--finalize` 74533ffbe, the formal gate runner 34e175824)
merged fast-forward with `codex/fr13-closure-safe-batch-20260812` (c3e5ea454: derived
closure counts f962c089f, launcher `:-1` fallbacks fbfa75c73, offload preflight retry
ce469053e, B1 sidecar git-free 569134c34).

This is the first schedule-front rung of the B4 pivot. It is where mamba narrowing's
retired throughput claim is supposed to reappear as real aggregate TPS: narrowing's
citable value is CAPACITY (KV peak 18% vs 74%), and capacity only converts to throughput
if something keeps the batch full.

---

## 1. THE PREMISE THIS RUNG WAS OPENED ON IS WRONG

The rung was scoped as "measure `FR13_B4_TASK_REFILL` ON vs OFF; refill keeps the batch
full, so events/step should climb toward 4."

**`FR13_B4_TASK_REFILL` does not change admission timing.** Its own docstring says so —
`scripts/run_swe_bench_q36_a.py:3253-3256`:

> WHAT IS ACTUALLY DIFFERENT FROM ex.map. ThreadPoolExecutor.map already queues every
> task up front and lets each worker pull the next one on return, so worker-level
> backfill is not new.

The dispatch confirms it (`scripts/run_swe_bench_q36_a.py:10142-10159`). Both branches
run the same `_job` over the same ordered `instance_ids` through a `ThreadPoolExecutor`
with `max_workers=concurrency`:

* OFF: `ex.map(_job, instance_ids)` — 16 futures queued up front, 4 workers, each worker
  pulls the next id the instant its previous job returns.
* ON: `_run_task_pool_with_refill` — a `deque`, 4 submitted at a time, next admitted on
  `FIRST_COMPLETED`.

Same FIFO order, same slot count, same instant of backfill. What the flag actually adds,
per its docstring items 1-3, is (1) completion-order result collection with a circuit
breaker, (2) **an admission ledger**, (3) a hard `peak_depth <= concurrency` invariant.
It is an EVIDENCE-AND-FAILURE-SEMANTICS flag, not a scheduling lever.

**The scheduling lever is the POOL SIZE, and it always was.** exact4 is the degenerate
case where pool == slots: the wave decays 4→3→2→1 with nothing behind it and the batch is
full width only ~36% of the arm. Any pool larger than the slot count backfills — with or
without the flag. The 2026-08-09 refill diagnostic that measured events/step 1.62→1.88
was comparing a **16-task pool against exact4**, not ON against OFF.

### Consequence for the comparison design

A pool16 refill-ON vs pool16 refill-OFF pair would spend ~10 GPU-hours measuring a null
that is readable from 20 lines of Python, and the OFF half would produce **no admission
ledger at all**, so it could not even be validated as a pool run. That arm is therefore
NOT RUN. This is a deliberate deviation from the rung's brief; the null is asserted
structurally instead, by test
(`tests/test_fr13_b4_pool16_refill_timing.py::test_refill_flag_is_evidence_not_a_schedule_lever`).

`FR13_B4_TASK_REFILL=1` is still **contract-pinned ON** for this class — because the
ledger is the only artifact that witnesses the occupancy claim, and
`fr13_floor_gate.validate_task_refill_ledger` already gates it fail-closed. Read the pin
as "instrumented", not "accelerated".

The honest comparator for pool16 is **exact4**, and exact4 is already sealed, citable, and
four passes deep on both topologies (§4).

---

## 2. THE CLASS

`run_class = pool16_refill_timing`, classification token
`real_swe_verified_pool16_b4_refill_timing`, schema `fr13.b4_pool16_refill_timing.v1`.

Distinct from BOTH neighbours, deliberately:

* NOT `real_swe_verified_exact4_b4_formal_floor_gate`. Different task set, different
  admission topology, different bracket mathematics. It is not a formal floor gate and
  carries no cap verdict.
* NOT the exact16 QC gate. Mark's 2026-08-10 ruling stands: exact16 is AGENT QUALITY
  CONTROL at batched-optimization milestones, not per-lever speed measurement. This class
  measures speed on the 16-task pool and asserts nothing about agent quality; the QC
  behavioural band remains a separate gate over the same subset.

### Binding

| | |
|---|---|
| subset | `config/fr13_fixed32/subset_b4_sixteen.json`, sha256 `47b0a3c9…dc0b5c` (= `EVIDENCE_SETS[16]`) |
| tasks | 16, the canonical astropy set |
| slots | `SWE_CONCURRENCY = MAX_NUM_SEQS_OVR = 4` |
| batch | `BSIZE = 4`, 32 physical rows, 31 drafts |
| contract-pinned stack | `FR13_B4_TASK_REFILL=1` (ledger evidence) |
| shipped defaults | resolved at the campaign commit from `scripts/fr13_canonical_env.sh` — narrowing `FR13_MAMBA_SPEC_BLOCKS_CDIV=1` is the default and is measured, not pinned |
| bracket topology | **`staggered`** required (envelope `max(post) − min(pre)`) |
| work census | MANDATORY — `fr13_measure.py:1743` already refuses a staggered reduction without it |
| pool ledger | REQUIRED — schema `fr13.task_refill.summary.v1`, `slots==4`, `task_count==16`, `completed==16`, `aborted==false`, `peak_depth<=4`, `time_weighted_mean_depth >= 3.2`, `full_width_fraction >= 0.60` |
| topologies | Tail23 (`tail6_fixed32`) and Hydra27 (`hydra27_fixed32`), both, every pass |
| passes | exactly 4 or 16 included, so the repo's pinned one-sided t critical applies |

### What it claims

Aggregate TPS, events/step and per-request step TPS delivered by the shipped fixed32 B4
stack when it is fed a **16-task pool at slot width 4 under staggered admission**, with
between-pass one-sided Student-t intervals.

### What it does NOT claim

1. **No exact4 comparability as a like-for-like.** Different tasks (16 vs the first 4),
   different bracket reduction (envelope vs widest-nested). Any exact4 contrast is
   descriptive and its confounds are enumerated in the artifact.
2. **No cap verdict.** `b4_cap_applicable=false`, unchanged: 35-38k contexts ⇒ 42-49 GB
   mandatory bytes ⇒ 155-179 ms floor, above the 137.607 ms one-sided cap.
3. **No agent-quality claim.** Resolve rate is recorded, never gated. The refill
   diagnostics resolved 9/16; that is the QC gate's business, not this one's.
4. **No claim that the refill FLAG produced the number.** §1.

### Primary statistic — and why it differs from the exact4 gate

The exact4 gate demotes `measured_tps_fullstep_wall` to "co-residency-dominated" and makes
`per_request_step_tps` primary, because at a 4-task pool the aggregate is a property of
task-end skew rather than of the stack.

**This class inverts that, on purpose and only within itself**: co-residency is exactly
what a pool16 arm is measuring, so aggregate TPS is the primary statistic and
`events_per_step` is the reported mechanism. The demotion of the aggregate at exact4 was
never a claim that the aggregate is meaningless — it was a claim that at pool == slots it
measures the wrong thing.

The 3c6d663d6 lesson (a candidate posted +17.2% aggregate while every request got 3%
SLOWER) is retained as a hard companion, not discarded: `per_request_step_tps` is reported
with its own full interval and, whenever an exact4 reference is supplied, an explicit
`per_request_non_regression` verdict is emitted alongside the aggregate. **A pool16
aggregate gain with a per-request regression is not a win**, and the artifact says so in
the payload rather than leaving it to the reader.

Note the class does NOT assume its own aggregate CV is lower than exact4's 17-27%. That is
a hypothesis (§3); the campaign measures it and the artifact reports it.

---

## 3. STATISTICS AND HOW MANY PASSES

### The variance we are fighting

From the sealed exact4 ON gate (4 included passes per topology, `T95_ONE_SIDED[3]`):

| statistic | Tail23 | Hydra27 |
|---|---|---|
| per_request_step_tps | 22.38 [20.34, 24.43] CV 7.8% | 21.20 [19.99, 22.41] CV 4.9% |
| measured_tps_fullstep_wall | 33.85 [28.67, 39.03] CV 13.0% | 34.47 [27.76, 41.18] CV 16.6% |
| events_per_step | 1.529 CV **19.3%** | 1.637 CV **20.0%** |
| step_wall_ms | 271.7 CV 6.0% | 279.8 CV 6.8% |
| apc_hit_rate | 89.8% | 91.2% |

The pass-level `events_per_step` values are 1.123/1.729/1.763/1.500 (Tail) and
1.680/1.886/1.818/1.162 (Hydra). One unlucky trajectory draw dominates a whole exact4 arm.

The expected effect is +15% to +20% on events/step (prior: 1.62→1.88 on the 2026-08-09
diagnostic, which ran narrowing-OFF with APC collapsed to 40.5%; narrowing is now the
default and KV peak is 18%, so the pool should thrash less and the effect could be larger).
**At exact4's 20% CV that effect is not separable at any affordable N.** The design has to
buy its power from the pool itself.

### Why pool16 should be tighter — stated as a hypothesis, measured as a result

At exact4 the drain-to-zero happens once and its duration is set by a single worst-case
trajectory. At pool16 the same drain happens once at the very end, but it is now amortised
over 4× the served work, and mean occupancy is an average over 16 trajectories rather than
4. Both effects push the between-pass sd of events/step down; a √N argument alone gives
~2×. If that holds, CV lands near 10% and:

| design | SE(mean) | one-sided t | detectable at 4 passes |
|---|---|---|---|
| CV 10%, n=4 | 5.0% | 2.3534 (df 3) | ~11.8% |
| CV 20%, n=4 | 10.0% | 2.3534 (df 3) | ~23.5% |

So a 15-20% effect is detectable at n=4 **iff** the tightening hypothesis holds. If it does
not, the artifact will say the interval does not exclude the exact4 band, and that is the
honest answer rather than a wider claim.

### Prior evidence, re-reduced offline before spending any GPU time

The one COMPLETE banked 16-task pool run — `fr13_b4_refill46_diag_20260809T040341Z`,
hydra27, 46 GiB KV, **narrowing OFF**, source 5e82859ae, self-labelled `citable=0` — was
re-reduced through the current `fr13_measure.py deploy-speed` with its work census:

| | pool16 diagnostic (non-citable) | exact4 ON gate (sealed, citable) |
|---|---|---|
| bracket topology | staggered, 13 distinct origins, census gate PASS | nested, 1 origin |
| events/step | **1.9934** | 1.529 (Tail) / 1.637 (Hydra) |
| aggregate TPS | **43.68** | 33.85 / 34.47 |
| per-request step TPS | 21.91 (derived) | 22.38 / 21.20 |
| step wall | 297.8 ms | 271.7 / 279.8 ms |
| census steps by batch width | 1:6042, 2:1789, 3:1151, 4:2408 | — |

That is +22% events/step and +27% aggregate at flat per-request — the exact profile a real
schedule win has, from a run that did NOT have narrowing's KV headroom. It is one draw at
an old commit under a different KV pool size and is cited here only as the prior that
justifies paying for four passes, never as a result.

It also validates both `fr13_measure.py` fixes on real data: the artifact now carries the
envelope basis string, and its `summed_bracket_inflation` spans 1.98-3.82x across 18
counters — the inflation that was previously withheld from staggered artifacts.

Two caveats it also supplies: 6042 of 11390 steps still ran at batch width 1, so a
16-task pool at 4 slots does NOT saturate the batch; and this run's occupancy cleared the
3.2 depth floor by only 0.034.

### Pass count: 4, not 2

Two passes give df=1, and `T95_ONE_SIDED` pins only df ∈ {3, 15}. The formal reducer's
rule — "the gate admits exactly 4 or 16 included passes" — is kept, unmodified and
unextended. **A 2-pass pool16 campaign is a screen and the reducer will say
`NOT_EVALUATED_INSUFFICIENT_PASSES`, `citable=false`.** No df=1 critical is invented to
paper over a short campaign.

Every pass is self-contained evidence and the reducer is offline and idempotent, so the
campaign can be read at any point and stopped early at the cost of citability only.

### Balance and atomicity — reused, not reimplemented

* `FR13_FLOOR_ORDER` alternates TH/HT by pass-index parity, so first-arm/second-arm
  position cannot alias into the topology contrast (`fr13_b4_formal_floor_gate.sh:134`).
* Pass atomicity (`fr13_b4_floor_gate_reduce.apply_pass_atomicity`, 04f341633): if any arm
  of a pass is invalid, every arm of that pass is voided. Keys only on validity, never on a
  measured value.
* Stack-state agreement across all included arms, resolved from
  `scripts/fr13_canonical_env.sh` **at the campaign's own commit** (af2325482 precedent).
* Outlier policy: none. Passes are rejected with a recorded reason, never cleaned.

### GPU-hours

Measured arm walls: exact4 arms 8-84 min (median ~45); the 16-task refill diagnostics ran
10 721 s and 7 856 s of pool wall (~2.2-3.0 h) plus boot/teardown.

| | per arm | per pass (2 topologies) | 4 passes |
|---|---|---|---|
| pool16 refill-ON | ~2.4-3.0 h | ~5-6 h | **~21-24 GPU-h** |

First paired read at ~6 h (pass 0), screen at ~12 h (2 passes), citable at ~21-24 h.

Costs NOT paid, and what they would buy:

* pool16 refill-OFF comparator — ~21 h — structural null (§1). Not run.
* in-campaign exact4 anchor, 4 paired passes — ~13 h — removes the between-campaign
  confound of §4. Deferred; escalate only if the pool16 result lands near the exact4 band.

---

## 4. THE COMPARISON

Comparator = the sealed exact4 ON gate,
`output/fr13_b4_formal_floor_gate_20260811T041931Z/fr13_b4_formal_floor_gate.json`
(`PASS`, `citable=true`, 4 included passes per topology, source commit af2325482,
`measured_stack_state` `{FR13_B4_TASK_REFILL: "0", FR13_FULL_ATTN_KV_FP8: "0",
FR13_MAMBA_SPEC_BLOCKS_CDIV: "1"}`).

The reducer accepts it via `--exact4-reference` and emits the contrast under an
`exact4_contrast` key that carries its own confound list. It is **descriptive**. Confounds,
all recorded in the payload:

1. **Task set.** 16 tasks vs the first 4 of the same canonical ordering. Trajectory length
   differs per task and events/step is trajectory-driven.
2. **Bracket mathematics.** Envelope (staggered) vs widest-nested. Both are census-gated —
   `fr13_measure` cross-checks each against `fr13_fixed32_work_census.jsonl` and refuses on
   disagreement — so neither is an unwitnessed sum, but they are not the same estimator.
3. **Between campaigns.** Different day, different source commit (c3e5ea454 vs af2325482).
   The intervening commits are launcher fallbacks, offload preflight retry, derived closure
   counts and a B1 sidecar fix; none touch the served stack. Verified by requiring the two
   campaigns' `measured_stack_state` to be equal — the contrast is refused if they differ.
4. **Bracket-origin count.** exact4 nested arms have 1 distinct origin; pool16 staggered
   arms had 13 on the diagnostic. Recorded per arm.

### The read this rung is for

* events/step ~1.53/1.64 → ~1.9 at flat per-request ⇒ ~+16-18% aggregate. That is the first
  real B4 win and the moment narrowing's capacity finally pays.
* events/step flat ⇒ pool depth is not the binding constraint at 4 slots; the next rung is
  oversubscription (8-10 sessions) or setup prefetch, and this rung is a recorded null.
* aggregate up, per-request down ⇒ **not a win** (3c6d663d6), reported as such by the
  `per_request_non_regression` verdict.

---

## 5. IMPLEMENTATION

Fail-closed, and every artifact the class demands is one the run already writes — checked
against the five "unsatisfiable precondition" fossils of this campaign (efdc728b5,
45d5b91d6, and the three that followed).

| artifact demanded | written by | verified present in |
|---|---|---|
| `deploy_speed_fullwall.json` (staggered, census-gated) | `--finalize` re-runs `fr13_measure.cmd_deploy_speed` with `--work-census` | produced by the reducer itself; census path checked before the call |
| `logs/fr13_fixed32_work_census.jsonl` | serving path, `FR13_FIXED32_WORK_CENSUS=1` in the canonical sequence | both refill diagnostics (81.7 MB) |
| `swe_out/verified/fr13_task_refill_summary.json` + `…_ledger.jsonl` | `_run_task_pool_with_refill`, only when the flag is 1 | both refill diagnostics |
| `container_env.txt` | serving path | every banked arm |
| `metrics_before_swe.txt` / `metrics_after_swe.txt` | serving path | every banked arm |
| `arm_ended_at.txt` | serving path | every banked arm |

The two occupancy floors are satisfiable and were derived from real runs — but note the
margin: the diagnostics measured depth 3.399 and 3.234 against a 3.2 floor, and full-width
0.780 and 0.710 against 0.60. **A pass can legitimately fail on the depth floor**, which is
the intended fail-closed behaviour (a pool that did not hold width is not a pool run) and
is recorded as an exclusion reason rather than a crash.

Changes:

1. `scripts/fr13_measure.py` — two staggered-provenance honesty fixes. The
   `bracket_reduction.basis` string fell through to the nested wording for staggered arms
   (claiming "the widest last-closing nested bracket" when the envelope was used), and
   `summed_bracket_inflation` was emitted only for nested, hiding the 1.87-4.00× refill
   inflation from the measure-side artifact. Both predate this rung.
2. `scripts/fr13_b4_floor_gate_reduce.py` — generalised to a RUN CLASS registry.
   `exact4_formal_floor` is the default and its behaviour is unchanged; `pool16_refill_timing`
   adds the 16-task binding, the staggered topology requirement, the pool-ledger gate, the
   inverted primary statistic and the optional exact4 contrast.
3. `scripts/fr13_b4_pool16_refill_gate.sh` — the campaign runner, structurally the
   formal gate runner with the pool16 binding.
4. `tests/test_fr13_b4_pool16_refill_timing.py`.
