# q1v3 v2 — disclosed post-hoc re-scoring (2026-10-02)

**Status.** This is a post-hoc analysis. It was defined after the frozen verdict and after the data had been seen, and it uses no new GPU runs. The pre-registered verdict is unchanged: **INCONCLUSIVE (R6)**. Under that rule the state tolerances were set at 4 × the worst native-pair distance over all calibration cells (1.05–1.59 relative L2). At that level 9 of the 10 injected faults go undetected.

**Data.** Run `q1v3v2-20261002T132534Z`:
- 147 forced-path cells (7 prefixes, cycles 0–6);
- 5 arms: the native reference A, three native comparison arms (B: process repeat; V: candidate KV capacity; P: stock packed GDN decode kernel), and the candidate;
- 10 negative-control observations.

Script: `scripts/v2exp/q1v3_posthoc.py`. Output: `POSTHOC_percell.json`.

## 1. State surfaces: compared at the same cell (GDN state, conv taps, new and historical KV)

**Rule.** For each cell and each surface, the reference is the largest native distance at that cell:

`R = max(B, V, P distance to A at this cell, 0.02)`, where 0.02 is the pre-registered floor.

The candidate's ratio to R is compared with each fault's ratio to R at its mutation cell (base case C2-nc-base, cycle 1).

| Surface | Worst clean-candidate ratio (147 cells) | Where |
|---|---|---|
| GDN state | 1.14 | cal-medium C5 k=0 (0.140 vs 0.123) |
| conv taps | 1.29 | cal-medium C5 k=0 (0.201 vs 0.156) |
| KV new positions | 1.17 | cal-long C2 k=2 (0.257 vs 0.219) |
| KV history | 1.13 | cal-medium C5 k=1 (0.205 vs 0.182) |

| Injected fault (mutation cell) | cal-short ratio | cal-medium ratio |
|---|---|---|
| NC_CONV (conv gather skipped) | 6.24 | 5.61 |
| NC_GDN (GDN replay skipped) | 2.83 | 3.73 |
| NC_KV (off-by-one KV remap) | 4.38 | 5.42 |
| NC_SIB (sibling rows committed) | 3.14 | 3.59 |

**Separation.**
- Every clean candidate cell is at most **1.29×** the native-kernel distance at the same cell.
- Every injected state fault is at least **2.83×**.
- Any multiplier in (1.29, 2.83) passes all 147 candidate cells and detects all 8 state-fault observations.

## 2. Next-token logits: compared by distribution

A per-cell bound does not work for KL. At many cells the native KL is close to 0, so 6 clean cells exceed 2× their per-cell native value. KL is therefore compared across all 147 cells.

| Arm vs A | Median KL | 90th-percentile KL | Max KL | Greedy-token flips |
|---|---|---|---|---|
| **LumoTree** | **0.043** | **0.127** | **0.491** | **16 / 147** |
| P (stock packed GDN decode kernel) | 0.043 | 0.095 | 0.490 | 16 / 147 |
| B (same configuration, repeat process) | 0.000 | 0.044 | 0.158 | 2 / 147 |
| V (candidate KV capacity) | 0.000 | 0.000 | 0.000 | 0 / 147 |

- **Candidate flips:** all 16 occur where the reference's top-two margin is at most 1.19 logits. At 8 of them a native arm also flips.
- **Stale-root-input fault (NC_STALE):** its KL is 1.66 on cal-short and 0.78 on cal-medium. Both exceed every clean candidate cell (max 0.49) and every native cell.

The other state surfaces show the same pattern as KL:

| Surface | LumoTree median / p90 / max | P median / p90 / max |
|---|---|---|
| GDN state | .186 / .240 / .303 | .183 / .240 / .275 |
| conv taps | .212 / .260 / .369 | .207 / .259 / .341 |
| KV new positions | .227 / .278 / .353 | .225 / .278 / .397 |

## Statement this supports

This is a disclosed post-hoc analysis, not the pre-registered test. Corrected 2026-10-02 after review: an earlier wording overstated both points below.

> Over 147 forced multi-cycle cells on prefixes up to 60k tokens, LumoTree's post-commit recurrent, convolution and attention state stays within 1.29x the distance between two native vLLM GDN decode kernels at the same cell. Each of the 8 injected state-fault observations (NC_CONV, NC_GDN, NC_KV and NC_SIB on two prefixes) is at least 2.83x. The 2 stale-root-input observations (NC_STALE) are logit faults. They exceed every clean candidate and native cell in next-token KL, but no per-cell logit rule was validated.

**What the logit table does not show.** Similar median, 90th-percentile and maximum KL, together with equal greedy-flip counts, do not establish that the output distributions match. That needs a sampling-level test (target and draft probabilities, and acceptance and correction sampling under the deployed filters) whose sensitivity to faults is shown first.

The pre-registered comparator could not certify equivalence, because its tolerance was too loose for the faults to register.
