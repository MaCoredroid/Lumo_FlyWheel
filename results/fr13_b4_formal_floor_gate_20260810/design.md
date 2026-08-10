# FR13 B4 formal statistical floor gate — design

Branch `codex/b4-twom-dual-blockmap-fix-20260807` (the B4 two-M dual-blockmap line).
Written 2026-08-10 against HEAD `0ebf2f8ba`.

The campaign log's open item is *"formal Tail/Hydra statistical floor gate — all pairs
so far are screens."* This document says what a CITABLE B4 gate run is, what already
exists to build it from, and — honestly — what trajectory variance does to an aggregate-TPS
claim at exact4.

---

## 0. The headline scouting result: the gate mostly already exists

The single most important finding is that **the formal statistical floor-gate machinery is
already written and already covers B4**. It was never run at B4 on the current stack. The
gap in the campaign log is a *missing run*, not missing code.

| Piece | Where | State |
|---|---|---|
| Canonical two-topology arm sequence | `scripts/fr13_fixed32_floor_timers_seq.sh` (`run_variant tail6_fixed32_$TAG` + `hydra27_fixed32_$TAG`, ordered by `FR13_FLOOR_ORDER=TH\|HT`) | exists |
| Campaign driver + gate invocation | `scripts/fr13_b4_campaign_driver.sh:226-238` → `$RUNROOT/fixed32_floor_gate.json` | exists |
| B4 statistical model | `scripts/fr13_floor_gate.py::b4_arm_statistics` (`:6265`), `moving_block_means` (`:6232`) | exists |
| Subset byte-binding, exact4 **and exact16** | `fr13_floor_gate.py::validate_canonical_subset` (`:1658`), `EVIDENCE_SETS` (`:127`) | exists |
| Topology-safe bracket reduction | `fr13_floor_gate.py::outer_counter_point` (`:6186`) — *union of counter indices, `task-sum is forbidden`* | exists |
| Per-arm phase decomposition + promotion rule | `scripts/fr13_b4_timing_math.py::phase_breakdown` / `promotion_verdict` | exists |
| Screen-grade per-arm reducer | `scripts/fr13_measure.py deploy-speed` → `deploy_speed_fullwall.json` | exists |

So the gate runner is thin: drive the canonical sequence at `BSIZE=4 CONC=4` on the pinned
exact4 subset, repeat it, and add the one thing genuinely missing — **a B4 aggregate-TPS
verdict with between-pass confidence bounds**, because the built-in verdict is a *cap*
verdict and the B4 cap is physics-dead (below, §4).

### 0.1 What is genuinely missing

1. **A TPS verdict layer.** `fr13_floor_gate.py` emits `gate_verdict ∈ {PASS, FAIL,
   NOT_EVALUATED_INVALID_INPUT}` against the *legacy SLO* (`u95(excess_ms) <= 0` vs the
   137.607 ms one-sided cap). At B4 that is a foregone FAIL — B4 mandatory bytes at
   ~35-38k tok/request are 42-49 GB ≈ 155-179 ms of pure weight streaming, i.e. the cap is
   *below the physics floor*. A B4 gate must report aggregate TPS + per-request TPS with
   bounds and treat the cap verdict as descriptive only.
2. **Between-pass repetition.** Both built-in uncertainty models are *within-run*. Neither
   sees trajectory variance, which is the dominant term for aggregate TPS (§3).
3. **A citability class label.** There is no enum; every sibling script hardcodes
   `formal_floor_acceptance_eligible: false` and defers to "the canonical statistical
   Tail23/Hydra27 floor campaign". This gate is that campaign, so it is the first artifact
   entitled to set it `true`. (Note the existing spelling drift to avoid propagating:
   `citable_cutlass_timing` vs `citeable_cutlass_timing`, and
   `formal_floor_acceptance_eligible` vs `floor_acceptance_eligible`.)

---

## 1. What a citable B4 floor-gate run consists of

### 1.1 Arms

Both topologies on the fixed32 contract (31 physical drafts + implicit root = 32 rows):

| Arm | Mode | Active drafts | Valid mask |
|---|---|---|---|
| `tail6_fixed32_$TAG` | `tail6_fixed32` | 23 (Tail23) | `0x7a9ce7ff` |
| `hydra27_fixed32_$TAG` | `hydra27_fixed32` | 27 (Hydra27) | `0x7abdffff` |

Both arms run the **current branch default stack**. No lever is flipped.
Recorded explicitly in the verdict:

- `FR13_MAMBA_SPEC_BLOCKS_CDIV` — **default `0` (OFF)** on this branch
  (`scripts/fr13_canonical_env.sh:44`, `fr13_launch_forked_fa2_tree_server.sh:2848`,
  `fr13_required_tree_flags.sh:55`: *"QUEUED 2026-08-09, default OFF"*). The narrowing
  flip awaits Mark; this gate measures narrowing-OFF and says so.
- `FR13_B4_TASK_REFILL` — **default `0` (OFF)**. Mandatory: the driver states refill output
  *"is NOT exact4-citable without a contract update"* (`fr13_b4_campaign_driver.sh:38-41`).
- `FR13_DRAFT_VOCAB_ROOT=1`, `FR13_DRAFT_VOCAB_K=65536` → weight floor **119.658 ms**
  (`fr13_fixed32_floor_timers_seq.sh:107-110`). This is the floor the brief pins.

### 1.2 Workload

- Subset: `config/fr13_fixed32/subset_b4_four.json`, byte-pinned
  `sha256 0e37b713…853f5` (`EVIDENCE_SETS[4]`). Tasks: `astropy__astropy-12907`,
  `-13033`, `-13236`, `-13398`.
- `BSIZE=CONC=4` (the sequence hard-requires `CONC==BSIZE`), `WALL=0` (no agent wall —
  long-tail tasks legally run 3h+; hang protection is the 600 s stall watchdog).
- Sampling: temp 0.6 / top_p 0.95 / top_k 20 / presence 1.0 / min_p 0 (`DEPLOY_FORCE_TEMP=0.6`).
- Reasoning-only assistant turns count as served (contract validator fix `c9c115f6f`).

### 1.3 Repeat structure

**4 independent passes per topology.** A pass = one full driver invocation running both
arms back to back, into its own fresh runroot.

Why exactly 4: `T95_ONE_SIDED` is pinned to `{3: 2.35336…, 15: 1.75305…}`
(`fr13_floor_gate.py:529`). Four passes give `df=3` and let the between-pass interval reuse
the *already-pinned* critical value rather than introducing a new constant. It is the
smallest N the repo's own statistics vocabulary admits. (16 passes is the next legal step;
it is affordable but not worth it — §3.4.)

Passes are **not** interleaved across topologies within a pass beyond the sequence's own
TH/HT order; alternate `FR13_FLOOR_ORDER` between passes (TH, HT, TH, HT) so first-arm/
second-arm position is balanced across the 4 passes and cannot alias into the topology
contrast (the second arm of a pass inherits a warmer page cache and a differently-aged
host).

### 1.4 Per-pass validation (all fail-closed)

Every pass must clear all of the following or it is *excluded and reported as excluded* —
never silently dropped, never repaired:

1. **Bracket topology.** At `B=4` with refill OFF all four tasks are admitted on one engine
   state, so the per-task `/metrics` brackets are **nested**, not disjoint.
   `fr13_measure.py::_bracket_reduce` (`:1487`) must classify `nested` and reduce to the
   widest (last-closing) bracket. A `disjoint` classification at B=4 is a red flag: on this
   branch a *staggered* arm misclassifies as disjoint and is only caught downstream by the
   census cross-gate, which fails closed (see §5 caveat).
2. **Work-census cross-gate.** `--work-census <arm>/logs/fr13_fixed32_work_census.jsonl`
   is mandatory. The census is a whole-arm engine-side record of completed pure-decode
   forwards and is topology-blind, so it is an independent witness. `cmd_deploy_speed`
   hard-fails when `bracket steps/events != census steps/events`
   (`fr13_measure.py:1706-1715`). This is the gate that would have caught the
   nested-summation bug that once inflated aggregates 1.7-2.6×.
3. **Floor-gate ingest.** `fr13_floor_gate.py` must return `analysis_valid: true`. Its own
   B4 evidence requirements must all hold: `MIN_B4_EXACT_EVENTS=512`,
   `MIN_B4_GE3_FRACTION=0.65`, `MIN_B4_MEAN_OCCUPANCY=2.9`,
   `MAX_B4_MEAN_OCCUPANCY_GAP=0.25` (cross-arm occupancy match),
   `MIN_TASK_COUNTER_STEPS=64`, `MIN_RETAINED_WALL_FRACTION=0.99`,
   `REQUIRED_COVERAGE=1.0`.
4. **Admission ledger.** Ingress/admission ledger present and self-consistent; peak
   admission depth ≤ `CONC`; no censored wall intervals (`wall_rejected == 0` —
   `fr13_floor_gate.py:4954-4962` raises on any censoring).
5. **Timing identity reconciliation.** `fr13_b4_timing_math.phase_breakdown()` must
   reconcile every arm record: SFWD event/step units, wall event/step units,
   `committed = accepted + 1`, `wall_tps == committed_per_event / wall_s_per_event`,
   `floor_ratio == wall_ms / floor_ms`, and the batch-invariant identity
   `per_request_step_tps == wall_tps / events_per_step`.
6. **Provenance.** Runtime manifest at-launch == at-end; external manifest at-launch ==
   at-end; subset sha256 matches `EVIDENCE_SETS[4]`; container env records the expected
   `FR13_FIXED32_MODE`, mamba/refill flag states, draft-vocab root/K.

**Outlier policy: there is none, deliberately.** The B1 gate has no trimming, winsorizing,
IQR fence, or z-score rejection anywhere; instead it *forbids censoring* and demands
`REQUIRED_COVERAGE = 1.0`. This gate copies that posture exactly: **reject a pass, never
clean it.** A pass that fails any check above is excluded with its reason recorded, and if
fewer than 4 passes survive the verdict degrades to `NOT_EVALUATED_INSUFFICIENT_PASSES`
rather than reporting a narrower-N interval.

---

## 2. Statistical acceptance

Two uncertainty models, reported side by side, because they answer different questions.

### 2.1 Within-run (existing, free, tight) — step wall and floor ratio

`b4_arm_statistics` runs a **moving-block bootstrap** over the per-step time series:
`BLOCK_SENSITIVITY = (64, 128, 256, 512)`, `BOOTSTRAP_REPS = 10_000`,
`BOOTSTRAP_SEED = 20260729` — reps and seed are *pinned and non-overridable*
(`fr13_floor_gate.py:7345-7349` raises if you pass anything else). The gated statistic is
the **worst (max) U95 across the four block lengths**, a centered 0.95 bootstrap quantile.
Block sensitivity is the autocorrelation defence: decode steps are strongly serially
correlated, and a naive i.i.d. bootstrap would understate the interval badly.

This model is sound for **step wall (ms/step) and floor ratio**, which are per-step
quantities with thousands of samples per arm. Those come out of one pass already citable.

### 2.2 Between-pass (new, expensive, wide) — aggregate and per-request TPS

Aggregate TPS is **not** a per-step quantity. It is
`aggregate_tps = events_per_step × per_request_step_tps`, and `events_per_step`
(co-residency) is a property of *one realization of four agent trajectories*. The
within-run bootstrap resamples blocks of a single trajectory draw; it cannot see the
variance between draws. **Reporting the bootstrap CI as the uncertainty on aggregate TPS
would materially understate it.** That is the central honesty point of this design.

So aggregate TPS gets a between-pass interval over the 4 pass-level values, using the
repo's pinned critical value:

```
point = fmean(pass_values)                     # mean, matching cluster_summary
sd    = stdev(pass_values)                     # sample sd, n-1
se    = sd / sqrt(4)
t     = T95_ONE_SIDED[3] = 2.3533634348018264
l95   = point - t*se        # conservative LOWER bound — the citable number
u95   = point + t*se
```

`cluster_summary` (`:6036`) is reused verbatim for `point/sd/se/u95`; the lower bound is
the same pinned critical applied on the other side. Reported for both
`measured_tps_fullstep_wall` (aggregate) and `per_request_step_tps` (batch-invariant),
per topology.

**The citable claim is the L95 on aggregate TPS**, i.e. "≥ X TPS at 95% one-sided
confidence", not the point estimate.

### 2.3 Promotion criterion is unchanged

If this gate is ever used to compare two stacks, `promotion_verdict()` applies unchanged:
`per_request_non_regression AND aggregate_gain`. That rule exists because the two-M arm
once posted +17.2% aggregate while per-request fell 2.96% — the whole gain was
co-residency the scheduler happened to supply, not faster service. The between-pass
intervals make that rule *testable* rather than a single-draw comparison.

---

## 3. Trajectory variance — the honest part

### 3.1 Measured spread on repeats of an identical config

The cleanest repeat set is the stock-CUTLASS Hydra27 arm: four runs, byte-identical
binaries, identical subset, identical env.

```
measured_tps_fullstep_wall = [49.77, 57.91, 34.95, 31.13]
n=4  mean 43.44  sd 12.56  CV 28.9%  max/min 1.86x
```

**That 86% run-to-run spread on identical config is larger than any candidate effect ever
measured on this rig.** The same two-M candidate against the same stock produced +6.5%,
+33.1% and +46.3% in three separate pairs — the candidate's apparent effect sits entirely
inside the stock arm's own noise band. (Two of the four values above, 49.77 and 57.91,
also predate the topology-safe reducer; restricting to post-fix arms narrows the set to
28.13/31.13/31.90/34.95, CV ~9%. Either way the conclusion below is unchanged, and the
gate excludes pre-fix arms automatically because they carry no work-census-gated
`bracket_reduction`.)

The mechanism is visible in the decomposition:

| Statistic | values across the four repeats | CV |
|---|---|---|
| `measured_tps_fullstep_wall` | 49.77, 57.91, 34.95, 31.13 | **28.9%** |
| `events_per_step` | 1.914, 2.247, 1.393, 1.475 | 25.4% |
| **`per_request_step_tps`** | 26.01, 25.77, 25.10, 21.10 | **8.9%** |

`prefill_frac` swings 0.310 → 0.942 (3.0×) and APC hit rate 0.561 → 0.928 across nominally
identical arms. `prefill_frac_note` in the emitted JSON already warns: *"MATCH it before
comparing aggregates across runs."*

### 3.1.1 Consequence: the primary statistic is the per-request rate

**Aggregate TPS is not a hardware metric on this workload — it is a co-residency metric,
and co-residency is set by agent trajectory.** Direct evidence from the orchestrator logs:
all four tasks are admitted at the same instant and then finish 745 s to 2484 s apart;
`astropy__astropy-12907` takes 406 s to 1419 s (3.5×) across arms, and
`astropy__astropy-13236` flips resolved↔failed between repeats of identical config. The
refill commit `28984c5db` states it outright: *"the alignment study attributes 54.7% of the
shortfall to task-end skew. With a pool of exactly four tasks that is structural… the
served batch is full width for only ~36% of the arm before decaying 4 → 3 → 2 → 1 with
nothing behind them. No kernel change can recover that."*

So the gate's **primary statistic is `per_request_step_tps`** (CV ~9%), with aggregate TPS
reported in full — it is the number the campaign cites — but labelled
`co_residency_dominated` and carrying its own much wider interval. This is not a new
judgement call: the repo already made it in `3c6d663d6` *"bind the B4 promotion criterion
to the per-request rate"*, after a candidate posted +17.2% aggregate while every individual
request got 3% slower.

Step wall is likewise stable (254.8-267.2 ms, CV ~2%), which is why the existing within-run
bootstrap is sound for it and unsound for the aggregate.

### 3.2 Wall-time variance — corrected

The "Tail23 arms ranged 52 min to 3 h on identical config" figure does not survive checking.
Enumerating every `tail6_fixed32` arm with start/end stamps across all worktrees:

**a single Tail23 arm is 27.6 min → 71.3 min (2.6×), median ~45 min.**

The "52 min to 3 h" conflates two real but different quantities: 52.6 min is exactly the
Tail23 m128-gate arm in stack `20260801T220208Z`, and ~2 h 48 m is the *Tail23 segment of
that same campaign* (three arms back to back: all-parent gate, m128 gate, timing stock —
span 2 h 47 m 50 s). The only literal ~3 h arm on record is a **16-task refill** Hydra
diagnostic (185.8 min, `arm_wall_s: 10720.65`) on a different config.

This materially reduces the cost estimate in §3.4.

### 3.3 What 4 passes actually buys

Half-width of the between-pass interval, by statistic and pass count:

| Statistic | CV | n=4 (t=2.353) | n=16 (t=1.753) |
|---|---|---|---|
| **`per_request_step_tps`** (primary) | 8.9% | **±10.5%** | ±3.9% |
| `measured_tps_fullstep_wall` | 28.9% | ±34.0% | ±12.7% |
| `step_wall_ms` | ~2% | ±2.4% | ±0.9% |

So 4 passes gives a genuinely useful **±10.5% on the per-request rate** and a frank
**±34% on the aggregate**. That is the honest picture: at exact4 the aggregate cannot be
pinned tightly at any affordable pass count, because its variance is structural to a
4-task pool, not statistical. Reporting it with a ±34% band is the correct thing to do —
it is precisely what stops the next screen from being read as a 30% win.

Practical consequences:

- A lever claiming **>15% on the per-request rate** is separable at 4 passes. The mamba
  narrowing screen (+19.4% per-request) qualifies.
- A lever claiming an aggregate-only gain is **not** separable at 4 passes and probably
  never will be at exact4 — which is the real argument for the 16-task pool (§3.5), not a
  reason to buy more exact4 passes.

### 3.4 Cost — corrected

One arm ≈ 27-72 min (median ~45 min). One pass = both topologies ≈ **1.3-1.8 h**.

| Plan | GPU-hours | per-request precision | aggregate precision |
|---|---|---|---|
| 1 pass | 1.3-1.8 | none (single draw) | none |
| **4 passes (recommended)** | **5.2-7.2, exp. ~6** | **±10.5%** | ±34% |
| 16 passes | 21-29, exp. ~24 | ±3.9% | ±12.7% |

**4 passes costs ~6 GPU-hours, not the ~14 first estimated** — the corrected arm walls make
this comfortably affordable and it should simply be run. 16 passes at ~24 GPU-hours buys
mostly a tighter bound on a statistic that is structurally noisy, and is not recommended;
if that precision is wanted, spend it on the 16-task pool instead.

### 3.5 The exact16 alternative, and why it is deferred

`EVIDENCE_SETS` byte-pins a 16-task subset (`subset_b4_sixteen.json`,
`sha256 47b0a3c9…dc0b5c`) and `validate_canonical_subset` accepts it, so **exact16 is
contract-legal, not a contract break** — it is even the driver's default `SUBSET`. Four
times the task diversity in one pass would attack trajectory variance at its source far
more efficiently than repeating exact4.

It is deferred for one concrete technical reason, not a policy one: with 16 tasks at
`CONC=4` and refill OFF, admission is **staggered/waved**, not one nested wave. On this
branch `_bracket_reduce` has only `nested`/`disjoint`; a staggered arm misclassifies as
`disjoint`, sums, and is then correctly killed by the census cross-gate. The **envelope**
reduction that handles this (`max(post) - min(pre)`) lives in
`scripts/fr13_b4_alignment_reduce.py`, added by `6877b4f1d` and extended by `d1fec8679`
— and **neither commit is an ancestor of this branch** (they are on `main` and eight other
branches). The campaign log already flagged this: *"Envelope bracket topology still needed
in fr13_measure for staggered arms (fail-louded correctly)."*

Porting is cheap and low-risk — `6877b4f1d` adds only two **new** files
(`scripts/fr13_b4_alignment_reduce.py`, `tests/test_fr13_b4_alignment_reduce.py`), zero
conflict surface, and the reducer is offline and read-only over recorded runroots. This
design ports the reducer as an independent cross-check on the exact4 gate, but does **not**
port `d1fec8679`'s `fr13_measure.py` hunk, because changing the in-run reduction would
change what the stack measures mid-campaign.

**Decision for Mark (not taken here):** exact16 would give ±5-6% on aggregate TPS for
roughly the cost of 4 exact4 passes, and it is subset-contract-legal. It requires porting
the envelope reduction into `fr13_measure.py` first. Recommended as the follow-up once the
exact4 gate has banked a citable baseline.

---

## 4. Why the built-in verdict is not the B4 verdict

`fr13_floor_gate.py`'s acceptance statistic is
`union_worst_block_legacy_slo_excess_ms_u95_le_0 AND exact_b4_worst_block_wall_u95_le_slo`
— the 1.15× / 137.607 ms one-sided cap. At B4 that is unreachable *by physics*, not by
engineering: measured context is ~35-38k tok/request, mandatory bytes 42-49 GB, weight-read
floor 155-179 ms > 137.607 ms cap. Measured B4 step walls are 254-267 ms ≈ 2.1-2.2× the
119.658 ms B1-basis floor.

Therefore the B4 gate:

- **runs** `fr13_floor_gate.py` and preserves its verdict verbatim as
  `legacy_cap_verdict` (descriptive), because everything else it validates — provenance,
  census, occupancy, coverage, union brackets, bootstrap — is exactly the rigor that makes
  a run citable;
- **decides** on aggregate TPS + per-request TPS with between-pass bounds;
- records `b4_cap_applicable: false` with the physics reason, so nobody later reads
  `gate_verdict: FAIL` as a regression.

---

## 5. Verdict JSON schema

Written to `<gate_root>/fr13_b4_formal_floor_gate.json`,
`json.dumps(..., indent=2, sort_keys=True, allow_nan=False)`.

```jsonc
{
  "schema": "fr13.b4_formal_floor_gate.v1",
  "analysis_valid": true,
  "gate_verdict": "PASS",         // PASS | FAIL | NOT_EVALUATED_INVALID_INPUT
                                  //      | NOT_EVALUATED_INSUFFICIENT_PASSES
  "formal_floor_acceptance_eligible": true,
  "citable": true,
  "classification": "real_swe_verified_exact4_b4_formal_floor_gate",

  "generated_at_utc": "…Z",
  "repo": "/…", "gate_root": "/…",
  "source_commit": "0ebf2f8ba…",
  "timing_harness_commit": "…",

  "contract": {
    "batch_size": 4, "concurrency": 4, "task_count": 4,
    "subset_path": "config/fr13_fixed32/subset_b4_four.json",
    "subset_sha256": "0e37b713…853f5",
    "task_ids": ["astropy__astropy-12907", …],
    "physical_rows": 32, "drafts": 31,
    "draft_vocab_root": 1, "draft_vocab_k": 65536,
    "weight_floor_ms": 119.658015414,
    "mandatory_weight_bytes": 32666638208,
    "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                 "presence_penalty": 1.0, "min_p": 0.0},
    "reasoning_only_turns_served": true
  },

  // exactly which stack ran — no lever was flipped for this gate
  "stack_state": {
    "FR13_MAMBA_SPEC_BLOCKS_CDIV": "0",     // branch default, narrowing OFF
    "FR13_B4_TASK_REFILL": "0",             // required for exact4 citability
    "FR13_FULL_ATTN_KV_FP8": "0",
    "kv_cache_memory_bytes": …,
    "defaults_flipped": []                  // MUST be empty for a citable gate
  },

  "b4_cap_applicable": false,
  "b4_cap_reason": "B4 mandatory bytes 42-49 GB => 155-179 ms weight floor exceeds the 137.607 ms one-sided cap; B4 is reported as throughput.",

  "topologies": {
    "tail6_fixed32": {
      "logical_topology": "Tail23", "active_drafts": 23, "valid_mask": "0x7a9ce7ff",
      "passes": [
        {
          "pass_index": 0, "runroot": "/…", "arm": "tail6_fixed32_…",
          "included": true, "exclusion_reason": null,
          "floor_order": "TH",
          "wall_s": 4831.2,
          "bracket": {"topology": "nested", "closing_task": "astropy__astropy-13398",
                      "distinct_bracket_origins": 1},
          "work_census_gate": {"status": "pass", "census_steps": 2919,
                               "census_events": 4740, "census_events_per_step": 1.6238},
          "admission_ledger": {"peak_depth": 4, "slots": 4, "wall_rejected": 0},
          "phase_breakdown": { … verbatim fr13_b4_timing_math.phase_breakdown() … },
          "measured_tps_fullstep_wall": 31.90,
          "per_request_step_tps": 21.91,
          "events_per_step": 1.456,
          "step_wall_ms": 267.21,
          "floor_ratio": 2.233,
          "prefill_frac": 0.53,
          "apc": {"queries": …, "hits": …, "hit_rate": 0.831},
          "legacy_cap_verdict": "FAIL",
          "within_run_bootstrap": {
            "model": "moving_block", "blocks": [64,128,256,512],
            "reps": 10000, "seed": 20260729,
            "step_wall_ms_worst_block_u95": 271.4
          }
        }
      ],
      "included_pass_count": 4,
      "excluded_pass_count": 0,
      "aggregate_tps": {                   // between-pass, cluster_summary + pinned t
        "cluster_count": 4, "df": 3,
        "point_estimate": 31.52,
        "sample_sd_across_passes": 2.81,
        "standard_error": 1.405,
        "t_0_95_one_sided": 2.3533634348018264,
        "l95": 28.22, "u95": 34.83,
        "cv": 0.0891
      },
      "per_request_step_tps": { … same shape … },
      "events_per_step":      { … same shape … },
      "step_wall_ms":         { … same shape … },
      "prefill_frac":         { … same shape … },
      "apc_hit_rate":         { … same shape … }
    },
    "hydra27_fixed32": { … same shape, valid_mask 0x7abdffff, 27 drafts … }
  },

  "comparison": {                          // descriptive, not a promotion decision
    "aggregate_tps_hydra27_minus_tail23": …,
    "intervals_overlap": true,
    "separable_at_95": false
  },

  "reference_points": {
    "screen_aggregate_tps_45_6": {
      "value": 45.6, "citable": false,
      "note": "two-M CANDIDATE arm with mamba narrowing ON; not the stock stack this gate measures"
    },
    "mamba_narrow_within_run_screen": {
      "per_request_tps_off": 15.07, "per_request_tps_on": 18.00,
      "apc_off": 0.831, "apc_on": 0.928, "citable": false
    }
  },

  "gates": {                               // every one must be true for citable:true
    "all_passes_nested_bracket": true,
    "all_passes_work_census_gated": true,
    "all_passes_census_agrees": true,
    "all_passes_timing_math_reconciles": true,
    "all_passes_floor_gate_analysis_valid": true,
    "all_passes_admission_within_slots": true,
    "no_censored_wall_intervals": true,
    "subset_bytes_canonical_exact4": true,
    "manifests_stable_launch_to_end": true,
    "no_defaults_flipped": true,
    "sufficient_included_passes": true
  }
}
```

`analysis_valid: false` + `gate_verdict: "NOT_EVALUATED_*"` is the fail-closed marker,
mirroring the B1 gate's deliberate distinction between "could not evaluate" and "failed".
Serialization uses `allow_nan=False` so a NaN raises rather than emitting invalid JSON.

---

## 6. Implementation plan

| Artifact | Role |
|---|---|
| `scripts/fr13_b4_formal_floor_gate.sh` | Gate runner. Preflight (`docker ps -aq == 0`, no orphan `[f]r13` pids, clean tracked worktree, GPU idle), then N passes of `fr13_b4_campaign_driver.sh` with `SEQUENCE_FILE=scripts/fr13_fixed32_floor_timers_seq.sh`, `BSIZE=CONC=4`, exact4 subset, alternating `FR13_FLOOR_ORDER`, each into a fresh runroot. Records the resolved stack state per pass. Runs detached under `setsid`. |
| `scripts/fr13_b4_floor_gate_reduce.py` | Reducer + verdict emitter. Offline, read-only over the pass runroots. Imports `phase_breakdown`/`promotion_verdict` from `fr13_b4_timing_math`, `cluster_summary`/`validate_canonical_subset`/`GateError` from `fr13_floor_gate`, and reuses `fr13_measure.py`'s bracket/census reduction rather than re-deriving any of it. Emits the §5 schema. |
| `tests/test_fr13_b4_formal_floor_gate.py` | Reducer tests on synthetic bracket fixtures, patterned on the existing `tests/test_fr13_deploy_speed_bracket_topology.py` helpers (`_write_bracket`, `_write_census`, `_concurrent_arm`). Required cases: **(a) nested-summation trap** — nested brackets whose naive sum inflates ~1.7-2.6×, reduced correctly to the widest bracket; **(b) census mismatch must fail closed** — bracket reduction disagreeing with the work census raises, never reports; (c) staggered admission fails closed; (d) fewer than 4 included passes ⇒ `NOT_EVALUATED_INSUFFICIENT_PASSES`; (e) a flipped default ⇒ `no_defaults_flipped: false` ⇒ `citable: false`; (f) NaN/non-finite anywhere ⇒ raise; (g) determinism — reduce twice, byte-compare (mirrors `fr13_floor_gate.py`'s `self_test`). |

No reduction mathematics is duplicated: bracket topology and census come from
`fr13_measure.py`, phase/rate identities from `fr13_b4_timing_math.py`, statistics and
subset binding from `fr13_floor_gate.py`, and the offline envelope cross-check from the
ported `fr13_b4_alignment_reduce.py`.

pytest invocation (mandatory house rules):

```
TMPDIR=/home/mark/shared/tmp-scratch .venv/bin/python -m pytest \
  --basetemp=/home/mark/shared/tmp-scratch/pytest-b4gate \
  --ignore=tests/test_codex_long_assets.py \
  tests/test_fr13_b4_formal_floor_gate.py
```

---

## 7. Open items requiring Mark's decision

1. **Pass count.** 4 passes ≈ 6 GPU-hours and is recommended and self-funding. 16 passes
   ≈ 24 GPU-hours buys a tighter bound on a statistic (aggregate TPS) whose variance is
   structural rather than statistical — not recommended; spend that budget on the 16-task
   pool instead.
2. **exact16 subset.** Contract-legal (byte-pinned in `EVIDENCE_SETS`) and statistically
   much more efficient, but needs the envelope bracket reduction ported into
   `fr13_measure.py` first. Recommended as the follow-up, not folded into this gate.
3. **Mamba narrowing stays OFF** for this gate, per the standing instruction not to flip
   defaults. Once Mark greenlights the flip, the gate reruns and the two verdicts form the
   citable promotion pair.

---

## 8. Addendum (2026-08-10, mid-campaign): two defects found by the first run

### 8.1 The driver's deploy-speed artifact is UNGATED

`fr13_b4_campaign_driver.sh::reduce` (`:185-199`) writes `deploy_speed_${TAG}.json` and
does **not** pass `--work-census`. Its output therefore records
`work_census_gate.status: "absent"` — an ungated B4 aggregate, exactly the artifact the
alignment study invalidated. The pair-runner
(`fr13_run_b4_cutlass_persistent_m128_timing.sh:613-617`) does pass it; the campaign driver
never did, because before this gate nothing downstream demanded it.

The gate correctly refused those arms. **Fix is offline and lossless:** every input the
cross-gate needs (the four per-task pre/post `/metrics` brackets and the arm's own
26.8 MB `logs/fr13_fixed32_work_census.jsonl`) is written during serving and persists, so
the reduction is simply re-run *with* the witness. Implemented as
`fr13_b4_floor_gate_reduce.py --finalize`, which calls the same
`fr13_measure.py cmd_deploy_speed` the driver called. Verified on pass_00: the aggregate is
**unchanged** (34.406 TPS both before and after) — finalization does not alter the
measurement, it *witnesses* it (census 3766 steps / 5206 events).

Finalization is idempotent, writes via `os.replace`, and refuses any arm lacking
`arm_ended_at.txt` — an arm still serving is still appending to its brackets and census, so
reducing it would be a torn read.

### 8.2 The canonical per-pass floor gate is unrecoverable for this campaign

`fr13_floor_gate.py:4879-4884` pins `summary.file_count == 62` and
`python_package_file_count == 25`. The fixed32 closure on this branch is **90 / 26**. Worse,
the pin contradicts the manifest builder itself: `fr13_runtime_manifest.py` PROFILES
`["fixed32"]` declares `package_file_count=26` (`:150`), so **25 is unsatisfiable by
construction**. The pin was last correct at `d944ae98b` (2026-07-30) and the closure has
since grown. `pass_00/fixed32_floor_gate.json` is the only such file anywhere in `output/`
— the canonical gate has never successfully run on this branch, so this is systemic and
pre-existing, not introduced by the gate runner.

**It cannot be repaired retroactively, and the reason is structural:**
`validate_source_fingerprint` rebuilds the closure and requires `current == end`, and
`scripts/fr13_floor_gate.py` is itself inside that closure (`FIXED32_VERDICT_TOOLS`,
`fr13_runtime_manifest.py:134-139`). So correcting the pin changes the closure hash and
makes the already-recorded manifests mismatch. **The pin is self-referential: you cannot
both match a recorded closure and have a corrected pin inside it.** Any fix therefore
applies to *future* campaigns only.

Consequence for this gate: **none that affects citability.** What the canonical gate would
have contributed is (a) its cap verdict, which at B4 is descriptive anyway
(`b4_cap_applicable: false`, §4), and (b) a within-run bootstrap on step wall, which the
between-pass interval on `step_wall_ms` covers. Crucially, **manifest stability is still
enforced** — by the driver's own `finalize_fixed32_manifest`, which byte-compares
at-launch against at-end and exits 14 on drift; pass_00 passed it. So the provenance
guarantee in §1.4 item 6 holds; only §1.4 item 3 (floor-gate ingest) is unavailable, and
the reducer never depended on it.

**Do not fix the pin while a campaign is serving.** Editing any closure file mid-flight
changes the closure between a pass's at-launch and at-end manifest, which trips
`finalize_fixed32_manifest` and kills that pass. The correction (62→90, 25→26, or better,
deriving both from the profile spec instead of duplicating them) must land after serving
completes.

### 8.3 First real numbers (pass_00, census-gated, narrowing OFF)

| Arm | agg TPS | per-request TPS | events/step | step wall | floor ratio | prefill_frac |
|---|---|---|---|---|---|---|
| Tail23 | 34.406 | 24.889 | 1.3824 | 268.95 ms | 2.126 | 0.532 |
| Hydra27 | 47.609 | 25.303 | 1.8816 | 291.86 ms | 2.307 | 0.505 |

This single pass already vindicates §3.1.1 empirically: the **per-request rate is
essentially topology-invariant (24.89 vs 25.30, +1.7%)** while the **aggregate differs by
38%** — and the entire difference is `events_per_step` (1.38 vs 1.88), i.e. co-residency.
Anyone citing the aggregate as a topology result would be citing the admission schedule.
Note also that Hydra27's higher aggregate comes with a *worse* step wall (291.9 vs
268.9 ms) and a worse floor ratio — it is not faster, it is more co-resident.
