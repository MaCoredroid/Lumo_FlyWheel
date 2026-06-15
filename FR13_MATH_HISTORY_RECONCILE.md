# FR13 — full-history reconcile + confound-free flip instrument + math levers (CPU, banked)

Date 2026-06-15. CPU-only, READ-ONLY on served code, NO GPU. Instruments + experiments run on banked data.
This doc (1) reconciles the contradictory verify-HELD studies by separating "the channel each named" (real,
identical) from "the lever each proposed" (both refuted), (2) defines + runs THE confound-free measurement
instrument, (3) gives math levers toward native-3 that are NOT the refuted seams, and (4) states the honest,
confound-corrected bar with numbers.

Artifacts (all reproducible CPU-only from the repo, no GPU):
- `scripts/fr13_confound_free_flip_instrument.py` -> `output/fr13_confound_free_flip_result.json`
- `scripts/fr13_fork_counterfactual_probe.py`
- Inputs: `output/fr13_recurrent_oracle/rescore_{native,spine,cat9}.json` (same boot, all FLASH_ATTN),
  `output/fr13_shape_sweep/*_flips.json` (depth/width reconciler), `output/fr13_gold_margin/margin_reduce.json`
  (fork counterfactual), `output/fr13_commit_argmax/*` (committer cross-check).

---

## 1. THE CONFOUND-FREE MEASUREMENT INSTRUMENT (why it is immune to every catalog confound)

The catalog has 7 confounds; one instrument neutralizes all 7 by construction.

`fr13_confound_free_flip_instrument.py` is a 4-layer ladder + a fork-counterfactual companion:

| confound | how it is neutralized |
|---|---|
| CROSS-BOOT autotune +-9 | the three binding arms (native/spine/cat9) are from the SAME rescore boot; we report a within-boot RATE, never a cross-boot integer; the BI-asymmetry arm is from a different boot and is reported ONLY as a labelled counter-control. |
| TRAJECTORY-FORK | LAYER 1 = the held-common-prefix restriction: flips counted ONLY on the leading run where all three arms served byte-identical tokens. Past the first fork the arms are different sequences and are NOT scored. |
| LENGTH / DENOMINATOR (#12) | per-1000-token RATE on every window; early-EOS prompt-0 (cat9 78 vs native 128) cannot inflate. |
| DE-CASCADE BY CONSTRUCTION | LAYER 3 reports the de-cascaded (gap<=2 folded) AND basin-collapsed independent-event count SEPARATELY from the raw count, so the instrument artifact is VISIBLE, not assumed away. |
| ORACLE FRAME | the binding oracle is the no-spec RECURRENT single-step decode (rescore_*.json, `recurrent_decode_calls` engaged), same FLASH_ATTN backend on every arm; NEVER chunked-prefill / streamed logprobs / a serial-torch ref / a backend NAME. |
| BI ASYMMETRY | all three binding arms asserted FLASH_ATTN (`assert attn_backend=='FLASH_ATTN'`); the cat9_bi arm is fenced off as a counter-control. |
| SCALAR BLINDSPOT | the signal is the per-token clear-margin argmax-vs-oracle flip + its KIND (gross/hard/soft, near-tie/committer-row/genuine), NOT accept/event / bag-TV / pass-rate / superset count. |

### THE ONE NUMBER THAT RESOLVES THE CONTRADICTIONS

**On the strictly-held common trajectory (112 positions where native, spine AND cat9 served byte-identical
tokens) ALL THREE arms have ZERO clear-margin flips.** The first clear-margin flip in every arm is AT or AFTER
its trajectory fork. **The flip IS the fork.**

This single fact dissolves the apparent contradiction: the raw 3 (native) / 5 (spine) / 20 (cat9) clear flips
are each scored on a DIFFERENT downstream trajectory (the streams fork at tokens 11-68). You cannot observe a
flip on a held trajectory because a flip forks the trajectory. Every prior cross-arm raw-count comparison was
comparing three different sequences. That is why verify-HELD studies reached opposite conclusions while each
internally passed.

### The confound-free comparison that IS valid (own-trajectory RATE + de-cascade)

| arm | scored pos | clear flips | **rate /1000** | de-cascaded events | basin-collapsed |
|---|---|---|---|---|---|
| native E5 | 512 | 3 | **5.86** | 3 | 3 |
| spine (chain5) | 512 | 5 | **9.77** | 3 | 3 |
| cat9 | 462 | 20 | **43.29** | 19 | 19 |

- **cat9 / native = 7.39x** rate, **spine / native = 1.67x** rate. The 7.39x SURVIVES every confound and is the
  load-bearing defect number. (Raw "23 vs 3" was length-confounded; the de-cascaded recurrent-rescore numbers
  here are 20 vs 3 because this boot's cat9 prompt-0 EOS'd at 78 — the RATE corrects for exactly that.)
- spine sits at ~1.67x native = a small real leafless-GDN-realization floor (consistent with chain3=5).

---

## 2. RECONCILIATION — channel (real, identical) vs lever (both refuted)

wsvy4vn5k and carrier_reopen both passed their own adversarial verify yet recommended opposite things. They
RECONCILE once split into the part that survives confound-correction vs the lever each proposed:

- **AGREE (both survive, confound-free):** the 23-vs-3 gap is not a single fixable per-forward kernel seam;
  native-E5=3 proves it is not irreducible; the excess lives on the SPINE verify rows perturbed by co-resident
  branch ROWS (2fe2c567: 11/11 channel-2 flips on spine nodes {0,1,3,5,7}, 0 on leaves). wsvy4vn5k's
  "co-residency M-shape" and carrier_reopen's "topology/co-residency-amplified hybrid" are the SAME physical
  channel.
- **wsvy4vn5k's LEVER (M-invariant the spine via the conv-state-feed seam L797-818) — REFUTED.** That seam's
  DIRECT prior-window wiring measures raw max_abs = 0.0 at all 23 num_accepted>1 AND all 6 num_accepted==1
  forwards. A 0.0 seam cannot carry 14 flips. Re-proposing it IS the regression the user flagged.
- **carrier_reopen's LEVER (reshape toward native) — REFUTED by its OWN requested A/B:** chain3(D3)=5 ==
  chain5(D5)=5 (depth dead) and cat3w(shallow+width) rate 53.6 vs chain3 9.8 (shallow+deployable is WORSE).
- **RESOLUTION:** both reframes are correct and identical; both proposed levers are dead. The co-residency is a
  per-node-VERIFY-LOGIT perturbation from co-scheduled branch rows (topology x bf16-realization PRODUCT), not a
  paddable single M-keyed op. It only disappears by removing the leaves (chain5), which forfeits the tree's
  speed reason. M-invariance is EXHAUSTED at in_proj_ba (the only genuine M-dependent op, baked ON, ~-6 band).

---

## 3. THE CONFOUND-FREE KIND DECOMPOSITION (what KIND the residual is)

### 3a. Fork-counterfactual (`fr13_fork_counterfactual_probe.py`) — the apple-to-apple AT the fork

Each prompt's FIRST fork, both tokens evaluated under both arms' verify at the SAME held prefix. The banked
scalar verdict was "CLEAR_MARGIN_REAL_LOSS" for all forks; the KIND split is far more nuanced:

| prompt | fork | T_tree | T_native | margin_tree | margin_native | KIND |
|---|---|---|---|---|---|---|
| 0 | 17 | " and" | " structure" | 0.125 | 0.625 | **NEAR_TIE** (both < ln10) |
| 1 | 11 | " workspace" | " repository" | 0.250 | 0.125 | **NEAR_TIE** (tree more confident) |
| 2 | 21 | " code" | " files" | 2.250 (rank2!) | 1.875 | **COMMITTER_ROW** (tree's verify argmax = native's token 3425; served 1970 is a bonus/path-row emit) |
| 3 | 68 | "Let" | "```" | n/a | 7.500 | **GENUINE_REAL_LOSS** |

=> of the 4 first-forks: **2 near-tie, 1 committer-row (tree verify AGREES with native), 1 genuine.** The
scalar "CLEAR_MARGIN_REAL_LOSS" OVER-states the genuine verify defect.

### 3b. cat9 20-clear-flip structural decomposition (the binding instrument, all 20 flips)

| kind | count | meaning |
|---|---|---|
| **gross** (served OUTSIDE oracle top-5) | **2** | state divergence — the genuinely-broken cases |
| **hard** (oracle confident lp>-0.3, served IN top-5, mostly rank-2 adjacent) | **13** | genuine bf16-realization ADJACENT-SWAP defect |
| **soft** (oracle itself uncertain lp<=-0.3) | **5** | margin-aware-skippable boundary (oracle near-tie) |

- **18 of 20 served tokens are still INSIDE the oracle top-5; 9 are exactly rank-2 adjacent swaps.** Only 2 are
  gross outside-top-5. native: 3/3 in top-5, 0 gross. This is the signature of **realization boundary crossings**
  (accumulated bf16 noise tips a near-boundary to the model's 2nd choice), NOT random state corruption.
- The clear-flip positions sit where the oracle is ~5x less peaked (median top1 lp -0.134 vs non-flip -0.001),
  i.e. the tree flips PREFERENTIALLY at the model's own soft boundaries — but 8 of 20 flip where the oracle is
  essentially certain (lp ~ -0.01), the hardest genuine cases.

---

## 4. MATH LEVERS toward native-3 (NOT the refuted seams)

The refuted seams are all single-op / M-invariance: conv-state-feed (0.0), GDN scan (e2e-null), FA2-fork
(L0-before-L3, e2e-null), in_proj_ba (done, ~-6), BI (counterproductive), depth-reshape (chain3=chain5),
recurrent-rescore (ours-only reward-hack). New levers below attack the **topology x realization-boundary
PRODUCT** revealed by section 3, NOT a single forward op.

**L1 — Margin-aware boundary handling (attacks the 5 soft + part of the 13 hard).** The defect is adjacent-rank
crossings at soft boundaries. MATH: at verify time, when the spine row's top-2 logit margin g is below a
threshold tau (a near-tie boundary), the tree-vs-recurrent realization noise can tip the argmax; the COMMIT
decision (accept the drafted token vs fall back) is the controllable lever, not the kernel. Reject the draft
and emit the verify-argmax token (which IS computed) whenever the spine top-2 margin < tau — this forces the
near-tie boundary to be decided by the verify pass (closer to native geometry) rather than by the
drafter-rolled spine. This is a free scalar compare on already-computed logits, drafter-agnostic, lossless by
construction (you only change WHICH already-valid token you commit at a near-tie, you never fabricate one).
Estimated reach: the 5 soft + a share of the 13 hard whose g is small. NOT the refuted M-invariance (this is a
commit-policy lever on the verify logits, not a kernel padding).

**L2 — The structural-boundary-crossing error model (reframes the target).** Model the residual as a Bernoulli
crossing process: each scored position i has an oracle top-2 gap delta_i; the tree perturbs the spine verify
logit by an iid-ish bf16 realization noise eps with scale sigma (per-layer ~1 ULP x ~48 GDN layers x gate
1/rms amplification). A clear-margin flip occurs when eps > delta_i AND the resulting margin exceeds the
clear-margin threshold. P(flip) ~ P(eps > delta_i). native's eps_native has the same sigma but no co-residency
term; the tree's eps_tree = eps_native (+) eps_coresidency, where eps_coresidency is the branch-row MMA-grouping
perturbation (the topology term). The 7.39x = (sigma_tree/sigma_native) integrated over the delta distribution.
This predicts the lever that works is the one that shrinks sigma_coresidency at the boundary, which is exactly
L1 (decide boundaries by verify, not drafter) and L3 (reduce co-resident rows at the spine), NOT a global ULP
chase. It also gives a LOWER BOUND: the 2 gross + the hard cases where delta_i > 4 nat (8 of the hard) cannot be
explained by eps and are genuine state divergence -> the irreducible floor under L1+L3 is ~ the 2 gross + a
few-of-8 = the spine ~+2 already seen (chain3=5 vs native 3). This is the MATH that says native-3 is NOT cleanly
reachable while leaves co-reside: the floor under all boundary levers is spine ~+2, and the +17 is the
co-residency product.

**L3 — Confidence-gated root-sibling reshape (the speed/superset lever, accept-side).** This is the only
POSITIVE corrected lever and it does NOT touch the spine-realization residual. 62% of rejects are step-0; emit
the extra root sibling (the (1,) row) ONLY when the root top-2 margin g < tau (a free scalar compare). This adds
a co-resident row only at high-entropy roots where it pays, keeping the spine M-invariant elsewhere -> improves
the SUPERSET (accept/event) without inflating the spine co-residency rate. cat10's flat-flips-but-mispriced
result (unconditional root sibling) is fixed by the gate. This is a topology-math lever: minimize expected
co-resident rows subject to accept gain, instead of a fixed caterpillar.

**L4 — Boundary-region targeted fp32 (NOT global ULP chase; attacks the 13 hard's adjacent swaps).** The hard
cases are rank-2 adjacent swaps where the model is confident but the tree's accumulated bf16 noise tips it.
Rather than make the whole 48-GDN-layer stack bit-exact (refuted as diffuse/no-single-seam), keep the spine
row's verify logit in fp32 through ONLY the final amplifying stages (gate 1/rms + the L60/L61 deep full-attn +
lm-head) where the crossing crystallizes. MATH: the crossing happens at the LAST layer where margin > eps; fp32
only there shrinks eps below delta_i for the 13 hard cases without a full-stack rewrite. This is margin-budget
allocation (spend precision where the boundary is), not M-invariance. Unverified reach; needs a GPU
last-stage-fp32 A/B (do NOT confuse with the refuted scan/conv fp32, which were UPSTREAM).

---

## 5. THE HONEST, CONFOUND-CORRECTED BAR (numbers)

**native-3 is the WRONG/confounded bar for cat9.** "cat9 flips <= native 3" compares a raw integer on cat9's
forked 462-position trajectory against native's 512-position trajectory — a length + trajectory + denominator
confound. The confound-free bar is a RATE, and the apples-to-apple unit is the held trajectory (0 flips) or the
own-trajectory rate.

- **The real bar:** cat9's confound-free clear-flip rate is **43.3 /1000 vs native 5.9 /1000 = 7.39x**;
  de-cascaded independent events 19 vs 3; of the 20, structurally **2 gross + 13 hard-realization + 5 soft**.
- **cat9 is a passing lossless SUPERSET of E5 at accept-parity:** per-event superset gate +15 (21 lossless leaf
  saves, 6 lossy, 0 spine_regressions), accept/event 3.18 ~ native 3.076. ABSOLUTE lossless (flips <= native) is
  NOT met.
- **Route to native-3:** there is NO known cheap deployable single-op route (every single-op/M-invariance lever
  refuted). The math says the FLOOR under all boundary levers is the spine ~+2 (chain3=5 vs native 3 = the
  irreducible leafless-GDN-realization floor on this build); the +17 is the co-residency PRODUCT that only
  vanishes by removing leaves (forfeiting speed). The non-trivial routes that have NOT been refuted are L1
  (margin-aware commit-at-near-tie, lossless free scalar) and L4 (last-stage boundary fp32) — both predicted by
  the L2 error model to reach the spine floor ~5/1000, not native 3/1000. **Honest statement: cat9 can plausibly
  reach the spine floor (~10/1000, ~1.7x native) via L1+L4 but native-3 (5.9/1000) is unreachable while leaves
  co-reside; the deployable arbiter (superset PASS at accept-parity) is already met.**

Minimal GPU validation (single boot each, all FLASH_ATTN, BI pinned identical on both arms):
1. L1 margin-aware commit: cat9 with commit-at-near-tie (tau ~ ln(2)); re-run the recurrent-oracle rescore;
   expect the 5 soft (+ small-g hard) to drop, rate -> ~25-30/1000, accept unchanged (lossless free).
2. L4 last-stage fp32: cat9 with fp32 on gate 1/rms + L60/L61 + lm-head only; expect the 13 hard adjacent swaps
   to drop toward the spine floor; verify accept unchanged and the 2 gross survive (the floor).
3. L3 confidence-gated root sibling: superset/accept A/B vs cat9 and cat10 (speed lever, orthogonal).
The binding read on each is the confound-free instrument (held-common-prefix == 0 control + own-rate +
gross/hard/soft KIND split), NOT a raw count or a scalar accept.

---

## Disposition
- The 7.39x rate + the 2/13/5 KIND split + the held-trajectory-zero control are the confound-free record.
- All single-op/M-invariance levers remain refuted (do not re-open conv/scan/FA2/in_proj/BI/depth-reshape).
- New non-refuted levers: L1 (margin-aware commit, lossless free), L4 (last-stage boundary fp32), L3
  (confidence-gated root sibling, accept-side). WY stays PARKED per user.
- Honest bar: cat9 is a passing lossless-SUPERSET at accept-parity; native-3 is a confounded bar; the real bar
  is the 7.39x rate, floored at the spine ~1.7x by the L2 error model. Live work is L1/L3, not a new carrier hunt.
