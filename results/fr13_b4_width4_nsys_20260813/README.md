# FR13 B4 — the width-4 nsys attribution, and what it does to the lever ladder

**DIAGNOSTIC. NOT CITABLE. `acceptance_valid=false`.** One pool16 arm was served
with CUPTI attached for its whole lifetime, so every absolute number here is
profiler-perturbed and none of it may be compared as a regression against an
unprofiled wall point. It is an attribution of *shares*, and the shares are what
the levers are priced against.

* capture — `output/fr13_b4_width4_nsys_20260813T030940Z/` (runroot + 187 MB
  trace + 548 MB sqlite, all on `/home/mark/shared`, all gitignored)
* tools — `scripts/fr13_b4_width4_nsys_{profile.sh,stepgate.py,reduce.py,table.py}`,
  `scripts/fr13_b4_batch_conditioned_wall.py`
* artifacts — `attribution.json`, `attribution_tables.txt`,
  `fr13_b4_batch_conditioned_wall.json` (this directory)

---

## 0. THE TWO RESULTS THAT MATTER

**(a) The width-4 operating point is 411.05 ms/step, not 387.59.** The sealed
number is a width blend. §5 below; it needed no GPU time and it revises the
denominator every lever is measured against.

**(b) The cost of width is FA2 and the GDN scan. It is not the GEMM.** Against
the B1 (batch-1) attribution, at width 4:

| component | B1 ms/step | width-4 ms/step | Δ | ratio | % of 411 ms wall |
|---|---:|---:|---:|---:|---:|
| target GEMM (fp8 CUTLASS) | 114.81 | 122.15 | +7.34 | **1.06×** | 29.7% |
| LM head (postprocess) | 12.35 | 12.67 | +0.32 | **1.03×** | 3.1% |
| **FA2 tree attention** | 21.37 | **69.75** | **+48.38** | **3.26×** | **17.0%** |
| **GDN scan (`_tree_gdn_path_kernel`)** | 12.36 | **40.98** | **+28.62** | **3.32×** | **10.0%** |
| GDN delta rule (cfwd) | 4.03 | 10.93 | +6.90 | 2.71× | 2.7% |

The GEMM and the LM head are **batch-invariant**: they re-read the same weights
whichever width the batch is, so 4× the requests costs them +6% and +3%. Every
millisecond that concurrency actually *costs* is in per-request work — attention
and the GDN scan — and those are the only two items on the board with enough
mass to move the wall.

---

## 1. CAPTURE VALIDITY — established BEFORE any kernel was named

`width4_window.md` §6 requires the capture to satisfy the census identity and
reconcile against the published split before a single kernel-level claim. The
reducer enforces that ordering in code: if the identity fails it emits the
validity section and withholds every kernel table.

| | |
|---|---|
| gate | absolute forward-step counter, **not** wall time (§6) |
| bracket | steps **[1790, 2329) = 539 steps**, edge ambiguity **0 steps** |
| armed at | step 1790, trailing events/step 3.824 over 200 steps |
| census identity | **539 census records == 539 counter steps** |
| event identity | **1760 census events == 1760 counter events** |
| NVTX cross-check | 541 `fr13.fixed32.step` instances vs 539 counter steps — inside the ±2 capture-boundary allowance the B1 reducer pins |
| batch widths in capture | `{1: 4, 2: 74, 3: 236, 4: 225}` — 85.7% of captured steps at width 3–4 |

Both identities are exact equalities, and they are the same ones
`fr13_measure.cmd_deploy_speed` enforces for a whole arm.

**Re-verified after the arm completed**, against the finished 67 MB census and
the published pool ledger rather than a mid-flight prefix — both identities still
hold exactly. The arm ran to `ARM_DONE ... swerc=0`, 16/16 tasks, 9/16 resolved,
`arm_wall_s` 4474.4, and its ledger reads `slots 4`, `task_count 16`,
`completed 16`, `aborted false`, `peak_depth 4`,
`time_weighted_mean_depth 3.206`, `full_width_fraction 0.619` — i.e. the pool
held width, so the captured steps sit inside a genuine depth-4 phase rather than
a drain.

### Split reconciliation against the sealed, unprofiled point

| component | capture | sealed (n=4 Tail23) | Δ% |
|---|---:|---:|---:|
| sfwd | 260.085 | 261.905 | **−0.7%** |
| dfwd | 50.202 | 45.534 | +10.3% |
| cfwd | 62.042 | 65.196 | −4.8% |
| **GPU component** | **372.329** | **372.636** | **−0.1%** |
| other (wall residual) | 19.220 | 14.952 | +28.5% |
| step wall | 391.548 | 387.588 | +1.0% |

The GPU component agrees to **0.1%**. That is the load-bearing check: it says the
capture is of the same object the sealed class measured. The profiler's cost is
visible almost entirely in the *wall residual* (+28.5%), which is where host-side
launch overhead belongs — so the GPU kernel attribution is usable and host-side
/ idle claims are the ones to treat carefully.

### CUPTI cost, measured rather than caveated

Conditioning the capture's own steps on width and comparing to the sealed
unprofiled width-4 wall:

```
width-4 wall, this capture (profiled)  428.52 ms/step
width-4 wall, sealed (UNPROFILED)      411.05 ms/step
=> CUPTI inflation                     +17.47 ms/step  (+4.3%)
```

**Reported, never subtracted.** No GPU total in this artifact has profiler cost
netted out of it; the B1 rule (locate the overhead, do not launder it) holds.
`PROFILER_OVERHEAD` totals 57.1 ms/step but sits on one flush thread (1373
records at 22.5 ms mean); every other thread is ≤0.009 ms/step — the same
thread-locality pattern B1 documented.

---

## 2. THE ATTRIBUTION TABLE

Per captured step, GPU rows only, divisor = 541 NVTX step instances.
`busy` is the **union** of `[start,end)` intervals, not the plain sum — the two
differ on this trace, so the union is used.

| range | span | busy | idle | GPU ops |
|---|---:|---:|---:|---:|
| **step (envelope)** | **391.590** | **373.071** | **18.519** | 2,761,464 |
| sfwd | 256.233 | 254.903 | 1.329 | 1,788,045 |
| cfwd | 61.951 | 57.857 | 4.094 | 631,642 |
| dfwd | 50.004 | 46.329 | 3.675 | 164,291 |
| postprocess | 12.670 | 12.670 | 0.000 | 1,080 |

The step envelope's span (391.590) matches the counter step wall (391.548) to
0.01% — an independent confirmation that the projection is measuring the step it
claims. Phase spans sum to 380.858, leaving **10.73 ms/step outside the four
phase ranges** (the post-DFWD tail), and phase idles sum to 9.098 against the
step's 18.519, leaving **9.42 ms/step of idle outside the phases**.

### Within SFWD (254.903 ms/step)

| group | ms/step | % sfwd | % of 411 ms wall | inst/step |
|---|---:|---:|---:|---:|
| **GEMM fp8 CUTLASS** | **122.151** | 47.9% | 29.7% | 255.8 |
| **FA2 attention** | **69.748** | 27.4% | 17.0% | 16.0 |
| **GDN scan** | **40.978** | 16.1% | 10.0% | 313.1 |
| elementwise | 10.633 | 4.2% | 2.6% | 1,239.7 |
| quant | 4.403 | 1.7% | 1.1% | 399.6 |
| gather/scatter | 2.127 | 0.8% | 0.5% | 313.1 |
| other | 1.917 | 0.8% | 0.5% | 425.0 |
| conv | 1.842 | 0.7% | 0.4% | 49.0 |
| gemm other | 0.602 | 0.2% | 0.1% | 47.6 |
| reduce/norm | 0.265 | 0.1% | 0.1% | 48.0 |

### Within CFWD (57.857) and DFWD (46.329)

| cfwd group | ms/step | | dfwd group | ms/step |
|---|---:|---|---|---:|
| elementwise | 28.447 | | gemm other (bf16 vocab head) | 16.876 |
| GDN delta rule | 10.928 | | unified attention | 15.589 |
| other | 8.672 | | GEMM fp8 (MTP) | 8.769 |
| sampling | 3.783 | | FA2 | 4.300 |
| softmax | 2.277 | | quant / other | 0.463 |

CFWD is **49% elementwise-and-other bookkeeping** (28.4 + 8.7 = 37.1 ms/step
across 321k tiny op instances) — the largest pile of small kernels in the step.

### Where the counter's `other` bucket actually goes

The counter split's `other` is a **wall** residual and is not the same object as
GPU idle or as the NVTX projection residual. Decomposed:

```
counter other_wall_ms_per_step        19.220 ms/step
  of which NVTX postprocess (LM head) 12.670 ms/step   (65.9%)
  remaining host/gap residual           6.550 ms/step
```

**Two thirds of `other` is the LM head — a GPU phase, not host gap.** Every
host-bookkeeping lever aimed at `other` is priced against **6.55 ms/step**, not
against 15–19.

---

## 3. THE LEVERS, PRICED AGAINST THE MEASURED TABLE

Denominator: the **true width-4 step wall, 411.05 ms** (§5). Detection
threshold at four passes: **MDE = 6.42 ms/step** (Tail23), 4.20 ms (Hydra27).

| # | lever | measured target | ceiling | vs MDE | verdict |
|---|---|---:|---:|---|---|
| 1 | **B4 FA2 `gqa_pair` re-test** | 69.75 ms/step (17.0%) | see below | ≫ | **RE-TEST — rank 1** |
| 2 | **GDN `single_launch` B4 route** | 51.91 ms/step GDN total | ~7.0 ms | 1.09× | **PROMOTE — rank 2** |
| 3 | **prefill contention** | **40.7%** of window wall is outside decode steps | dilution factor | n/a | **rank 3, and it taxes 1 and 2** |
| 4 | **F-window 4-byte D2H** | 6.55 ms host residual; 0.081 ms of ≤8 B copies | ~2.9 ms | 0.45× | **BELOW THRESHOLD** |
| 5 | *(new)* GEMM + LM head | 134.82 ms/step (32.8%) | **0** | — | **FLOOR, not a lever** |

### 1. FA2 `gqa_pair` — rank 1, and its B1 null does not transfer

FA2 is **69.75 ms/step, 17.0% of the width-4 wall**, and it grew **3.26×** from
B1 while the GEMM grew 1.06×. The lever was recorded null at eff 1.2 — but that
null was measured against a target **3.26× smaller**. A relative improvement that
was invisible at B1 is 3.26× more absolute here: recovering even **10% of FA2 is
7.0 ms/step and clears the MDE**; 20% is 14 ms and clears it by 2.2×.

B1's roofline put FA2 at 4.93× above its bandwidth floor (21.36 measured vs 4.33
floor, 17.03 ms of structural headroom). If KV traffic scales with batch the
width-4 floor is ~17.3 ms against 69.75 measured — **~52 ms of structural
headroom**, the largest single block of addressable time in the step.

*Falsification:* if a width-4 FA2 re-test moves `step_wall_ms` by less than
6.42 ms, the gqa_pair mapping is genuinely null at width 4 too and rank 1 dies —
but it must be re-tested at width 4 before that can be said, because the
published null was taken where the kernel was a third of its current size.

### 2. GDN `single_launch` — rank 2, and width is what makes it measurable

GDN is **51.91 ms/step** (40.98 sfwd scan + 10.93 cfwd delta rule), 12.6% of the
wall, and the scan runs **313.1 instances/step** at width 4 against 96.1 at B1 —
launch count scaled **3.26×**. The B1-proven saving was **−2.2 ms/step**; if it
is launch-bound it scales with launches to **~7.0 ms/step**, which clears the
6.42 ms MDE by 1.09×.

That margin is thin and honest: this lever is separable at four passes **only if
the gain really is launch-bound**. It is the cheapest promotion on the board
because the kernel work is already done.

*Falsification:* if the B1 gain was tile/occupancy-bound rather than
launch-bound, it will not scale, will land near 2.2 ms, and will be
unmeasurable at n=4.

### 3. Prefill contention — rank 3, and it taxes every decode lever

Observed directly and unmistakably during the capture: for **~10 minutes** the
pure-decode step counter did not advance at all (1153 → 1153) while
`prompt_tokens_total` climbed 4.04 M → 5.46 M with 4 requests resident.

Measured from the trace's own NVTX step ranges (540 real ranges; one
capture-boundary artifact at a bogus timestamp excluded):

```
collection extent            360.19 s
inside decode step ranges    213.75 s   (59.3%)
outside decode step ranges   146.44 s   (40.7%)
  concentrated in 33 gaps > 1 s totalling 146.4 s, longest 20.6 s
mean decode step duration    395.83 ms over 540 steps
```

**40.7% of the captured window wall is not inside a decode step at all**, and it
is not diffuse — essentially all of it is 33 discrete gaps averaging 4.4 s. Those
are prefill bursts and agent tool-use stalls, not per-step overhead.

This is not a kernel lever, and it is the reason the other two are worth less
than their ms/step suggests: a decode-step saving of X ms moves total wall by
roughly 0.59·X. It is also why `events_per_step` in the capture is 3.265 against
the window's 3.600 — co-residency oscillates with agent tool-use.

### 4. F-window 4-byte D2H — real, and below the detection threshold

The memcpy census prices it directly: **≤8-byte copies cost 0.081 ms/step of GPU
time** (72.3 copies/step); D2H copies overall are 6.32/step at 0.014 ms/step. The
cost was never the copy — it is the sync stall, which at B1 was 2.91 ms of GPU
idle. That stall is **one sync per step and therefore batch-invariant**, so it is
still ~2.9 ms at width 4 — **0.71% of the wall, against a 6.42 ms MDE.**

**It cannot be measured on its own at four passes.** The whole host-side residual
it lives in is only 6.55 ms/step. Bundle it with rank 2 or leave it.

### 5. The floor nobody can move — the finding that bounds the campaign

**GEMM + LM head = 134.82 ms/step, 32.8% of the width-4 wall, and both are
batch-invariant** (1.06× and 1.03× from B1). Against the published 126.514 ms
hardware floor, this *is* essentially the floor, already being paid.

So the arithmetic of the whole width-4 war: eliminating **all** of FA2 and
**all** of GDN — which nobody can do — leaves
`411.05 − 69.75 − 51.91 = 289.4 ms/step`, still **2.29× the 126.514 ms floor**.
The realistic ladder (rank 1 at 10–20%, rank 2 at ~7 ms) totals roughly
**14–21 ms/step, or 3.4–5.1%**.

That is the honest headline: **this ladder does not close the width-4 gap.** It
is worth running because 14–21 ms is separable and real, not because it gets the
stack near the cap.

---

## 4. RECOMMENDED ATTACK ORDER

1. **FA2 `gqa_pair` at width 4** — largest mass (69.75 ms), largest structural
   headroom (~52 ms), and its recorded null was measured against a 3.26× smaller
   kernel. Re-test before anything else.
2. **GDN `single_launch` B4 route** — kernel work already done, ~7.0 ms expected,
   clears the MDE by 1.09×. Cheapest promotion available.
3. **Prefill contention / scheduling** — 40.7% of window wall sits outside
   decode steps, in 33 discrete gaps, so a decode-step saving of X ms moves
   total wall by ~0.59·X. Not a kernel change; the biggest structural item left.
4. **CFWD small-kernel consolidation** *(new, unranked)* — 37.1 ms/step of
   elementwise + other across 321k tiny instances per step. Not yet a lever
   because no candidate exists, but it is the third-largest addressable pile and
   nothing has ever been aimed at it.
5. **F-window 4-byte D2H** — only bundled. Unmeasurable alone at n=4.

**Do not** spend width-4 GPU time on the target GEMM or the LM head. They are
batch-invariant weight traffic and they are the floor.

---

## 5. THE DENOMINATOR CHANGED: 411.05 ms, NOT 387.59

`width4_window.md` §3 named one gap as the largest remaining and declared it out
of reach:

> A true `batch == 4` filter is **not derivable from this evidence**: the census
> resolves batch width per step but carries **no per-step wall**.

True of the census, false of the evidence set. The SFWD timer's per-step samples
sidecar (`fr13.sfwd_per_step_samples.v2`), written beside **every** arm already,
carries `wall_fwd_indices` / `wall_ms` / `wall_drafts` — per-step wall *and*
per-step width, on the same absolute step index. Verified twice before use:

* the sidecar's per-step width equals the census `batch_size` on **all 9385**
  shared steps of pool16 tail23 pass 0, zero mismatches;
* the selected samples **are** the sealed counter bracket — 4868 samples summing
  to 1951.9978 s against the sealed bracket's 4868 / 1951.9978 s. The tool raises
  rather than reports if either disagrees.

| statistic | Hydra27 | Tail23 |
|---|---:|---:|
| sealed (rescaled blend) | 389.61 ms, CV 1.00% | 387.59 ms, CV 1.51% |
| direct blend (same samples) | 395.02 ms, CV 0.99% | 393.02 ms, CV 1.62% |
| **batch-conditioned, width 4** | **413.14 ms, CV 0.86%** | **411.05 ms, CV 1.33%** |
| **MDE at n=4** | **4.20 ms** | **6.42 ms** |

The +23.5 ms correction (+6.0% on both topologies) decomposes into two effects
that must not be collapsed:

* **basis, +5.4 ms** — the sealed `step_wall_ms` is a cross-population rescale,
  `(wall_s/wall_drafts) × (fwd_drafts/fwd_steps)`. That is the documented "do not
  share a basis" caveat, now quantified.
* **width, +18.1 ms** — the pool-depth window is a width blend: only ~70% of its
  wall-bracketed steps are width 4, and width-3 steps run ~50 ms cheaper
  (361.7 vs 411.05).

Conditioning also **dissolves** the basis caveat rather than measuring it: at
`width == 4`, events/step is exactly 4.0 by construction, so per-event and
per-step wall differ by exactly 4 and no cross-population rescale exists to
disagree about. And it **tightens**: CV falls on both topologies, so the MDE
improves to 6.42 / 4.20 ms.

Every lever in §3 is priced against 411.05 ms and 6.42 ms.

---

## 6. WHAT THIS DOES NOT CLAIM

1. **No acceptance or regression reading.** CUPTI was attached for the whole arm;
   the measured inflation is +17.47 ms/step at width 4 and is reported, never
   subtracted.
2. **No new sealed operating point.** §5 is a re-reduction of a non-citable
   instrument class and inherits `citable=false`. It revises no sealed verdict.
3. **No causal claim that width causes the extra wall.** Co-residency and context
   length move together across these arms; §5 selects steps, it does not
   randomise them.
4. **No claim that the capture is a random sample of the window.** It is 539
   consecutive steps at one point in one arm, and its `events_per_step` (3.265)
   sits below the window mean (3.600). Shares are stable across the split
   reconciliation; absolute ms/step are not a four-pass estimate.
5. **No throughput claim of any kind.** Every number is per-step cost.
6. **The B1 comparison is descriptive.** Different batch, different task set,
   different campaign; it is used for *scaling ratios*, which is what it can
   support, not for absolute contrast.

## 7. A STALE PREMISE, CORRECTED

The brief for this rung carried forward a "1.945 ms of phantom `cudaGraphLaunch`
cost that was pure CUPTI" from the B1 analysis. **No such finding exists.** The
only `1.945` in the tree is `1.945375655`, the B1 **wall/floor ratio**
(232.780 / 119.658 ms/step). `cudaGraphLaunch` appears once, as 2.554 **host CPU**
ms/step, and it was never subtracted from anything because the B1 method never
lets a host row enter a GPU total in the first place.

The real B1 CUPTI rule is stronger and is the one followed here: profiler
overhead is **located**, not subtracted — B1 published 38.8 ms/step of
`PROFILER_OVERHEAD` and showed it sat on a flush thread rather than the critical
CUDA thread. This capture reproduces that pattern (57.1 ms/step on one flush
thread, ≤0.009 ms/step everywhere else) and subtracts nothing.
