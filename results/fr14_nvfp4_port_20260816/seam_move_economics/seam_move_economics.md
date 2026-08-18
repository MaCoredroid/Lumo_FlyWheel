# Seam-move economics — FR14 K0 banked data

Read-only analysis at frozen HEAD `05987f682`. No GPU touched, no container started, nothing
written outside this scratchpad. Pre-registration in `PREREGISTRATION.md` (written before any
coverability number was computed). Raw outputs: `coverability_raw.json`, `coverability_deep.json`,
`seam_economics.json`, `frontier_K0.json`, `frontier_pooled.json`.

The original framing ("are the deep accepts suffix decoding?") was withdrawn mid-task and is
answered by construction — see §1. This file answers the replacement question: **what does moving
the MTP/suffix seam earlier cost and buy?**

---

## 1. The seam is architectural, not a discovery

`scripts/fr13_mtp_suffix_assembly.py:assemble_tail_tree(head_depth=5, mtp_k=5)`:
head = draft positions 0-4, spine tokens are **pure MTP** (`mtp_k == head_depth`, "byte-identical
to the baseline cat33333"), 2 MTP runner-up branches per depth; tail node j sits at path length
`head_depth+1+j` = **draft position 5+j**, spine-only, filled from Arctic
(`arctic_inference.suffix_decoding.SuffixDecodingCache`, hash-pinned at prelaunch,
`fr13_launch_forked_fa2_tree_server.sh:7048`).

Census of this serve agrees: `main_tail_length=6`, `arctic_requested_tokens=12`
(main 6 / rank1 4 / rank2 2), `active_nodes=23`, 31 physical drafts. Per-position accept counters
are non-zero exactly through position 10 = 5 + 6 - 1.

**Positions 0-4 are MTP; positions 5-10 are suffix.** The 0.597 survival at position 5 is the
MTP->Arctic handoff conditional; the code names it ("the tail's arctic top-1 conditional is
weakest at the handoff j=0/d6"). Deep accepts being copyable is the mechanism, not evidence.

## 2. Evidence inventory — what the banked outputs do and do not contain

HAVE:
- **Per-position accepted counters per task** (`vllm_metrics_pre/post.txt` deltas). 4 tasks in the
  K0 serve; per-task deltas sum exactly to the serve total (92439). Across `output/` there are
  **22 usable runs / 228036 steps / 936372 accepted tokens**, all tail6-family (max position 10).
- **Full agent trajectories** (`swe_out/verified/per_task/*/qwen_trace.jsonl`): the initial prompt,
  every tool result (the model's context), and every emitted assistant turn (thinking + text +
  tool-call arguments). This is the prompt/context/emission material.
- **Work census** (21611 step records) — drafter/committer/TAW routes, shapes, launch counts.
- **GPU timers** (`fr13_sfwd_sidecar`, `deploy_speed_b1radix.json`) and an **nsys profile with
  kernel-level attribution** (`fr13_fixed32_b1_nsys_20260818T001018Z/.../*.sqlite`).

DO NOT HAVE (this bounds every conclusion below):
- **No per-step accepted-length records.** The census carries `taw.accepted_lens_shape` etc. —
  shapes, not values. Grep over all census keys: no accepted-length value, no bonus-token value.
- **No token IDs anywhere.** No per-step emitted token stream, so no way to align an accepted run
  to the text it produced.
- **No per-token proposer attribution.** Nothing records "this accepted token came from MTP vs
  Arctic" beyond the positional convention of §1.
- **Nothing measured past draft position 10** in the entire corpus (checked `output/` and
  `results/`): zero runs with a longer tail.

Consequence: **exact per-accepted-token attribution is impossible.** The strongest available test
is a simulation of the suffix proposer against the reconstructed emitted stream, cross-validated
against the measured per-position ladder. That is what was run.

## 3. Pre-registered rule and its gates (outcomes)

Rule (verbatim in `PREREGISTRATION.md`): rebuild the token stream per task with the served
tokenizer; stand-in suffix proposer = longest earlier n-gram match (L in 24..2) -> most frequent
continuation, chained on its own proposals; `S(j)` = length of the correct chain from emitted `j`.

- **Ladder validation gate: PASS.** Required >=4 of 5 slots within +/-0.10.
  simulated r2..r6 = .780 .843 .880 .906 .921
  observed  (pooled 22 runs) = .795 .850 .878 .895 .909 -> max |delta| = **0.015**
  observed  (K0 serve)       = .803 .858 .888 .898 .907 -> max |delta| = 0.023
  5/5 slots pass, by a factor of ~5 more precision than the gate demanded. The stand-in reproduces
  the shipped Arctic tail's conditional survival curve. This is the load-bearing validation.
- **Reconstruction-fidelity gate: FAIL on 3 of 4 tasks.** Reconstructed / reported output tokens =
  0.963, 0.748, 0.747, 0.672 (pooled 73.3% of emitted mass). Per pre-registration the coverability
  numbers are therefore approximate. Diagnosis: the trace's `usage.input_tokens` implies roughly
  twice as many model calls as recorded turns, so the missing mass is almost certainly auxiliary
  CLI calls (compaction/summarisation), not missing trajectory content. Direction of bias unknown.

## 4. Measured suffix coverability

Pooled over 21985 sampled emitted positions (plus an independent 12000-position replicate):

| quantity | value |
|---|---|
| **q1 = suffix cold-start hit rate, unconditional** | **0.4756 / 0.4702 (replicate) -> 0.473** |
| q1 given prev 1/2/3/4/5 tokens were suffix hits | .782 / .848 / .884 / .906 / .920 |
| chain ladder r2..r6 (simulated) | .780 .843 .880 .906 .921 |
| chain ladder r7..r14 (simulated, **no GPU measurement exists**) | .930 .929 .937 .948 .947 .941 .961 .956 |
| P(S >= d): d=1,2,3,4,6,8,10,14 | .470 .367 .309 .272 .227 .196 .174 .143 |

Two things follow immediately:

1. **The suffix proposer is a poor cold proposer and an excellent warm one.** Cold: 0.473.
   Warm (slot >= 2): 0.78 rising monotonically to ~0.95. MTP is flat at 0.77-0.84 across all its
   depths. So suffix loses to MTP at any handoff and beats MTP everywhere after it.
2. **The measured 0.597 handoff at position 5 is the cold-start rate plus a selection premium.**
   Conditioning on 5 correct MTP tokens selects easier/more copyable regions: 0.473 -> 0.597
   (+0.124). A depth-3 handoff carries less selection, so
   **q1(seam 3) is bracketed [0.473, 0.597], central 0.548** (linear in the number of preceding
   MTP-correct tokens).

### Answer to (1): how much of the position 3-5 mass is suffix-coverable
(0-indexed draft positions; the brief's "pos 4-5" = positions 3 and 4.)

| position | MTP survival today (measured) | suffix survival if it took over | coverable share | novel / MTP-only |
|---|---|---|---|---|
| 3 (would be handoff slot) | .8083 | q1 = .473 / .548 / .597 | 59% / **68%** / 74% | 41% / **32%** / 26% |
| 4 (would be chain slot 2) | .8169 | r2 = .803 | **98%** | 2% |
| 5 (already suffix) | .5972 | becomes chain slot 3 = .858 | >100% (improves) | — |

**The non-coverable mass is concentrated entirely in the handoff slot.** Every slot after the
handoff is at least as good under suffix as under MTP. A seam move does not lose the deep spine —
it pays one handoff penalty, and pays it at a shallower depth where more of the step's probability
mass is still alive (cumulative survival .616 at position 3 vs .407 at position 5).

## 5. The cost side is now MEASURED, not modelled

Kernel attribution inside the `fr13.fixed32.dfwd` NVTX span (banked nsys sqlite, 40 sampled steps):

```
5.00 x cutlass::device_kernel          17.35 ms/step
4.00 x kernel_unified_attention_2d     11.20 ms/step
5.00 x nvjet_sm121_tst_mma_128x192x64   8.68 ms/step
10.0 x nvjet_sm121_tst_mma_64x112x64    6.87 ms/step
10.0 x cutlass::Kernel2                 4.71 ms/step
1.00 x flash::flash_fwd_splitkv         1.98 ms/step
                       TOTAL kernels   51.48 ms/step   (drafter NVTX span = 52.75 ms)
```

The drafter body repeats **exactly 5 times** per step (5x device_kernel, 5x nvjet-128x192,
5x gatherTopK, 5x silu; 10x = 2/pass; attention = 1 flash_splitkv + 4 unified). The drafter is
97.6% GPU kernels and essentially **all** of it is the 5 MTP passes — there is no large fixed
drafter overhead to worry about. Per-pass: pass 1 ~9.5 ms, passes 2-5 ~**10.3 ms** each.

This **confirms the briefed ~10.5 ms/pass by independent kernel attribution.** Dropping 2 passes
saves **20.6 ms** of a 207.87 ms step (sfwd 134.55 + drafter 52.674 + committer 20.642;
`derived_tps_fullstep_gpu` = 25.39 tok/s; wall basis 215.31 ms / 24.51 tok/s).

## 6. Answer to (2)/(3): the seam-3 trade

Equal-reach comparison (both configs reach draft position 10; tail 6 -> 8 to replace the 2 head
depths; nodes 21 -> 17, still inside the 31 physical slots, verify rows stay 32 so verify and
committer are held fixed).

**K0 basis (the definitive serve, today = 25.39 tok/s):**

| q1(seam3) | accept | step ms | **TPS** | vs today |
|---|---|---|---|---|
| .473 pessimistic (no selection credit) | 3.834 | 187.3 | **25.81** | +1.7% |
| .548 **central** (linear selection interp) | 4.066 | 187.3 | **27.05** | **+6.5%** |
| .597 optimistic (full depth-5 selection credit) | 4.221 | 187.3 | **27.88** | +9.8% |

**Pooled basis (22 runs, today = 24.56 tok/s):** 25.48 / **26.45** / 27.09 = +3.8% / **+7.7%** / +10.3%.

Sensitivity to the per-pass cost (9.5 / 10.3 / 11.2 ms) moves these by only about -/+0.4 tok/s.

**Answer as asked: seam-at-3 predicts ~27.0 tok/s against today's measured 25.4 (+6.5%), with a
credible range of 25.8-27.9 (+1.7% to +9.8%).**

Evidence class of every input:
| input | class |
|---|---|
| today's 25.39 tok/s, accept 4.2774, step decomposition | **MEASURED** |
| MTP survivals at positions 0-2 (.9472/.8443/.7709) | **MEASURED**; assumed invariant to head depth (same heads, same branches — safe) |
| suffix continuation ladder r2..r6 | **MEASURED** (today's tail; 22 runs agree within 0.02) |
| per-MTP-pass 10.3 ms | **MEASURED** (nsys kernel attribution) |
| q1 upper bound .597 | MEASURED at depth 5, on a **differently selected population** |
| q1 lower bound .473 | **SIMULATED**, on a validated simulator, on 73% of the emitted mass |
| q1 central .548 | **INTERPOLATED** — the single weakest number in the chain |
| r7, r8 (the 2 tail nodes that restore reach) | **SIMULATED ONLY** — nothing past position 10 has ever been measured |
| verify + committer unchanged | assumed; rows are fixed at 32 (measured). Any saving is upside |

**Pre-registered verdict: MARGINAL.** The rule required the pessimistic case (q1=.473, tail8,
central cost) to beat today by >=5% for FAVORABLE; it delivers +1.7%. The optimistic case delivers
+9.8%, which is the MARGINAL band. The move is more likely than not a real gain, but it is not
robust to its weakest assumption.

## 7. Answer to (4): coverability at positions 1-3, and whether the seam generalises

Suffix cold-start is 0.473 **everywhere** — it is a property of the text, not of depth. MTP
survivals at positions 1, 2, 3 are .844, .771, .808. So the suffix proposer is a strictly worse
cold proposer at every shallow depth; the seam question does **not** generalise on accept grounds.
It generalises only on *cost* grounds, because MTP passes are expensive (10.3 ms ~ 5% of the step).

The full frontier, equal reach at position 10, central cost, central q1 (K0 basis):

| seam n (MTP depths) | accept | step ms | TPS | vs today |
|---|---|---|---|---|
| 5 (today) | 4.278 | 207.9 | 25.39 | — |
| 4 | 4.178 | 197.6 | 26.21 | +3.2% |
| 3 | 4.066 | 187.3 | 27.05 | +6.5% |
| 2 | 4.024 | 177.0 | 28.39 | +11.8% |

The model says "keep cutting MTP passes", which is really the statement **each MTP pass is barely
worth its 10.3 ms**. Treat n=2 as a hypothesis, not a result: it rests on q1 extrapolated two steps
beyond any measurement and on the simulated r7-r9.

The cleanest low-risk experiment the data supports is not seam-3 at all but **seam-4**: drop ONE
MTP pass, extend the tail by one node. Predicted +3.2% (range -1.3% to +4.3%), and it tests the
q1-at-shallower-depth assumption directly at minimal blast radius.

## 8. The bigger lever in the same data: the tail is too short

Today's tree uses 23 active nodes of 31 physical slots — **8 free slots** — and the tail's
conditional survival is *still rising* at the tree edge (.907 at position 10; simulated .93-.95
beyond). Extra tail nodes cost no MTP pass and no verify row (rows fixed at 32).

| config (no seam change) | accept | TPS at unchanged 207.9 ms | vs today |
|---|---|---|---|
| tail6 (today) | 4.278 | 25.39 | — |
| tail8 | 4.495 | 26.43 | +4.1% |
| tail10 | 4.686 | 27.35 | +7.7% |
| tail12 | 4.857 | 28.18 | +11.0% |

And combined, at today's active-node count (n=3, tail14 = 23 nodes): predicted **30.9 tok/s
(+21.7%)** central.

Two honest caveats: (a) every one of these rows depends on the **simulated** ladder past position
10, which no GPU run has ever measured; (b) deeper trees lengthen the committer's tree walk
(`taw.loop_iterations=12`, `gdn.critical_path=12` today) and the Arctic token request, so the
20.6 ms committer is not guaranteed flat. But the *ranking* is robust: tail extension changes no
MTP pass and cannot regress accept (padded non-matching tail nodes are lossless by the monotone
committer, per the module's never-regress argument), whereas a seam move trades measured accept
for measured time.

Related: `scripts/fr13_tail_config_sweep.py` already encodes this exact model and assumes
`ARCTIC_PURE = 0.55` for the pure-suffix conditional. **This study measures that constant at 0.47**
— the existing sweep tool is ~15% optimistic about suffix-only drafting, which is precisely the
regime a seam move pushes into.

## 9. What this analysis cannot show

1. **No exact attribution exists.** No per-step accepted lengths, no token IDs, no proposer tags.
   Every coverability number is a validated simulation plus ladder arithmetic, never a direct count.
2. **73% emitted-mass coverage.** The pre-registered fidelity gate failed on 3 of 4 tasks; the
   missing ~27% is probably auxiliary CLI calls, but its copyability was not measured.
3. **Nothing past position 10 was ever measured.** Every "restore the reach" and "extend the tail"
   number depends on the simulated ladder.
4. **Run-to-run variance swamps the predicted effect.** The same arm (`tail6_fixed32_b1radix`)
   banked accept = 3.81 / 4.04 / 4.28 across three runs (+/-10%), and s5 ranges .496-.647 across
   the 22 banked runs. A predicted +6.5% sits inside that band: a seam-3 A/B must be paired on an
   identical task set with >=20k steps per arm, or it will measure task mix, not the seam.
5. **4 astropy tasks.** Copyability of SWE-bench Python-repo work need not generalise.
6. One assumption worth stating because it is favourable and load-bearing: for a deterministic
   top-1 proposer, P(proposal == sampled token) *is* the rejection-sampling acceptance probability,
   so the simulator's hit rate is an unbiased estimator of accept rate rather than a proxy — valid
   as long as the committer is lossless (Gate 1) and the emitted text is an unbiased target sample.

## 10. Bottom line

- Seam at 3: **~27.0 tok/s predicted vs 25.4 measured today (+6.5%; range +1.7% to +9.8%).**
  Pre-registered verdict **MARGINAL** — real but not robust to the one interpolated input.
- The whole cost of a seam move is **one handoff slot** (survival .81 -> ~.55). Everything deeper
  is at least as good under suffix as under MTP; 98% of position-4 mass is suffix-coverable.
- Suffix cold-start hit rate is **0.473** — measured here for the first time, versus the 0.55 the
  repo's own sweep tool assumes.
- The larger, lower-risk lever in the same data is **tail length, not seam position**: 8 free node
  slots, a still-rising tail conditional, +7.7% predicted for tail10 with no MTP change.
- Recommended order: (1) extend the tail at the current seam and *measure* the ladder past
  position 10 — this also retires the biggest simulated input; (2) seam-4 as the single-pass probe;
  (3) seam-3 only after (1) and (2) land. Any A/B must be paired on an identical task set.
