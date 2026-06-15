# FR13 — Speed-Tuning Plan, Branch B: TOPOLOGY RESHAPE (cat10 revive + remove-deep / add-root trees)

Date 2026-06-15. **READ-ONLY design** (no kernel/patcher edited; pathspec commit of THIS doc only).
Companion to `FR13_SPEED_TUNING_PLAN.md` (Branch A = OPT-1 G2 sync-kill). HEAD ~`8b2c6d3e` (3-lever wf:
OPT-1 sync-kill + OPT-A fp8 + this TOPOLOGY RESHAPE). GOAL (user): **cat9 B=1 decode-TPS STRICTLY >
native E5**, lossless gate HELD per-change. Branch B = the user's "revive cat10 w/ accounting attention +
remove-deep-add-root trees" lever.

Designs: (1) **cat10 revive, accounting-correct** (avoids the prior [2,6,8,6]/2.932 artifact); (2) the
**REMOVE-DEEP-LEAF / ADD-ROOT-LEAF candidate trees** (exact `tree_choices`, reusing the committed infra);
(3) per-candidate **SPEED + ACCEPT + lossless prediction** (depth-matched, INFERRED-labeled); (4) the
**accounting-correct measurement** that AVOIDS the cat10 artifact.

---

## 0. WHAT THE PRIOR RESHAPE CAMPAIGN ALREADY SETTLED (do NOT re-litigate)

Two GPU A/Bs ran. The RECURRENT-frame A/B (`FR13_RESHAPE_AB_RECURRENT_BIND`, wf_0e61765e, the
deployment-correct frame, verify HOLDS) is canonical:

| arm | shape | flips (recurrent oracle) | accept/event | note |
|---|---|---:|---:|---|
| native E5 (FLASH MTP-5) | depth-5 linear | **3** | 3.08 | the lossless BAR + speed BAR |
| cat9 (LOCKED) | depth-5 + 4 leaves | **23 (~18 de-casc)** | **3.198** | lossy, FAST (accept edge) |
| chain3 (ours) | depth-3 spine, NO width | **1** ≤ native | 2.266 | LOSSLESS, SLOW (leaf-free) |
| cat3w (ours) | depth-3 + root + d1 leaves | ~17 | 2.282 | lossy AND slow (worst) |

**Settled negatives (carry, do not overturn):**
- **DEPTH is +1; WIDTH/leaf-co-residency is the flip CARRIER (+16).** chain3 (d3, leaf-free)=1 vs cat3w
  (d3, +width)=17 ⇒ width +16; cat3w(d3)=17 vs cat9(d5)=18 ⇒ depth +1. The leaves are BOTH the accept
  edge AND the lossy co-residency — **coupled**.
- **Reshape EXHAUSTED for the LOSSLESS-FLIP axis** (FR13_RESHAPE_EXHAUSTED_BIND): topology alone gives a
  lossless shape (chain3) OR a fast shape (cat9), not both — removing leaves to cut flips removes the
  accept edge.

**⇒ Branch B is NOT a flip-reduction lever.** It is a **SPEED-with-lossless-HELD** lever: at the CURRENT
cat9 lossless operating point (held per change, not improved), find a reshape that makes decode-TPS > native
by trading the N_PAD / GDN state-row-traffic cost against accept/event. The 22-flip chase is the PARALLEL
drift front, out of scope here per the speed-first order (project_fr13_speed_first_lossless_gate).

---

## 1. THE SPEED ACCOUNTING THAT MAKES "REMOVE-DEEP / ADD-ROOT" A REAL LEVER

Two cost laws (FR13_SPEED_HISTORY_RECONCILE §"What is LEFT"; FR13_SPEED_TAX_BASELINE):

1. **lm-head verify rows are M-INVARIANT: +0.0019 ms per verify row** (539 rows / +1 ms). The verify
   lm-head gathers all M tree rows into ONE `[M,5120]·[5120,248320]` bf16 GEMM, weight 2.5428 GB read
   ONCE. Adding/removing a node's verify ROW is **nearly free at the lm-head** (the +81.9 ms tax FIX-1
   removed was the DRAFTER double-head, already gone). ⇒ "skip lm-head for unservable nodes" is ~a NO-OP;
   do NOT design around it.

2. **The binding per-NODE cost is GDN state-row traffic + the N_PAD STEP.** `n_pad = next_pow2(N+1)`:
   - N ≤ 7 → **pad8**  (chain5 N=5)
   - N = 8..15 → **pad16** (cat9 N=9: **+42-46 ms/fwd over chain5 = ~7× the 3N+2a+1 row-traffic floor**)
   The N=5→9 jump is dominated by the **pad8→pad16 step** (h_cache=(N_PAD,BV,DIM_K) register-bound; N_PAD=16
   is the only 0-spill cat9 geometry at BV=16/warps=8 — 254/255 regs; FR13_SPEED_HISTORY §N_PAD).

**The design key:** TPS = accept_tok / s_per_fwd. The cheapest way to push cat9 TPS over native is NOT
fewer lm-head rows (free) — it is to **keep N ≤ 7 (stay in pad8)** so s/fwd drops toward chain5's regime,
WHILE keeping enough accept (via a confidence-gated add-root sibling) that accept/event holds the
break-even. A pad8 tree that holds cat9-class accept beats BOTH native AND cat9 on TPS.

> Honest label: every per-forward ms is INFERRED (census/literature-anchored; nsys per-kernel export
> empty). The pad8/pad16 step + the M-invariant lm-head are MEASURED (ptxas wp5hsu63v / fdf5ffa7). The
> clean B=1 `decode_seconds` boot is the arbiter (feedback_dont_handroll_speed).

---

## 2. cat10 REVIVE — ACCOUNTING-CORRECT (the [2,6,8,6] / accept 2.932 was an artifact, not a loss)

### 2a. Why cat10's prior "accept 3.198→2.932 = wider hurts" was an ACCOUNTING ARTIFACT
cat10 = cat9 + the depth-0 root sibling `(1,)` (10 nodes, committed depth still 5):
`[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]`. The banked −0.27 is
**artifact-dominated** (FR13_CAT10_INVESTIGATE_BIND, holds=FALSE verify-CORRECTED;
feedback_check_artifact_before_concluding):
- **(i) Class-12 whole-window TRAJECTORY confound (dominant):** cat9 vs cat10 generated DIFFERENT greedy
  streams (diverge p0@17/p1@11/p2@21/p3@61); cat10 p0 hit EOS 25 tokens sooner (73 vs 98) ⇒ 25 fewer
  accepted tokens over ONE MORE event ⇒ mechanically lower accept/event. Cross-trajectory accept/event is
  NOT apples-to-apple — it is class-12 confounded, never a superset verdict.
- **(ii) The sharp d0→d1 drop is the SIBLING-STOP DENOMINATOR artifact**, NOT a co-residency m1 bug: a
  root-sibling win `[1,]` is `accepted_len=1` (caps at d0), swelling `per_pos[0]` while contributing **0
  to d1+**, deflating the d1|d0 conditional. De-confounding pos0 RECOVERS d1|d0 to ~0.84+; d2/d3/d4
  conditionals are FLAT cat9-vs-cat10.
- **(iii) m1 (verify co-residency) is STRUCTURALLY RULED OUT:** the verify `strict_mask` walks parents to
  root; node `(1,)` is never a spine ancestor ⇒ NO spine row has `strict[spine, root_sib]=1` — the
  root-sibling row is attention-INVISIBLE to every spine row (the GDN tree-scan uses the same mask).
  Residual = at most a sub-ULP fp reduction-order leak in the shared pad tile, NOT −0.27.
- **(iv) The d0 RESCUE is REAL:** d0 accept rate 0.871→0.906 (+0.035, ~+21/boot). The root runner-up is
  the truth **27%** of the time when root top-1 misses (a 2-horse-race near-tie signature; random rank-2
  ≈ 0%). The 62%-of-rejects-at-step-0 are exactly what the root sibling rescues.

**Net: cat10's real per-event yield ≈ cat9 + a small d0 rescue, once you strip (i) the trajectory
denominator and (ii) the sibling-stop denominator. The −0.27 was accounting, not a leaf-lossy loss.**

### 2b. How to MEASURE cat10 correctly (the confound-free protocol — §4)
Depth-MATCHED (cat10 committed depth 5 ⇒ **vs native E5**, never E3); PAIRED teacher-forced accept on a
byte-identical served prefix (NOT cross-trajectory whole-window); a **per-node sibling-vs-spine counter**
(FR10_METRICS=1 + a d0 sibling-win tag) so sibling-stop events can be removed from the `per_pos[0]`
denominator. This is the decisive cat10 test the prior boots LACKED (no per-node winner log was saved —
FR13_CAT10_INVESTIGATE_BIND "DECISIVE post-fix test").

### 2c. cat10 the RIGHT way = the FREE CONFIDENCE-GATED root sibling
Unconditional cat10 pays +1 verify row (free at lm-head, M-invariant) AND stays pad16 (cat9 is already
pad16) so no pad win — yet the d0 rescue fires on only ~3.5% of events. The directive-named lever
(FR13_CAT10_INVESTIGATE_BIND §"THE LEVER"; FR13_MATH_HISTORY:151): emit `(1,)` **only when the root is a
near-tie** (`margin = logit[rank1] − logit[rank2] < tau`, tau ≈ ln 2). The gate is **FREE** —
`top2 = torch.topk(_fr10_logits, 2)` is already materialized in the drafter (patcher :11167/:11178); one
scalar compare/event, zero extra forward. On confident roots serve clean cat9 (no sibling); on near-ties
add the sibling that pays. Keeps the +0.035 d0 rescue WITHOUT any unconditional dilution. **Shape-true:
caterpillar + at most ONE single root node, never deeper.** Class-9 note: a gated tree makes
draft-toks/event VARIABLE — the engagement gate must assert tok/draft ∈ {9,10}, NOT a fixed int (build it
as its own exact-match shape with a variable-count check, OR boot ungated cat10 first for the clean
number, then gate).

---

## 3. REMOVE-DEEP-LEAF / ADD-ROOT-LEAF CANDIDATE TREES (exact tree_choices, infra-reuse)

Principle (the §1 accounting + Sequoia/EAGLE-2/TALON, already cited in FR13_RESHAPE_SHAPE_DESIGN): **width
pays at SHALLOW depth, diminishing at deep; cut the deepest leaves (and the deepest spine node) to drop
into pad8, recover accept with a confidence-gated ROOT sibling.** The deepest leaves are ALSO the lossy
co-residency carrier — so removing them helps speed AND (held-not-improved) lossless.

All shapes = sorted `(len, path)` `tree_choices`. Downstream consumers (parent/ancestry masks, committer
path enum, eager-pack replay rows, conv-fusion prior windows) **AUTO-ADAPT off SPEC_CONFIG `tree_choices`**
(patcher :11000-11001); only the drafter PACKING is hand-rolled, and the existing `_fr10_spine_steps` /
`_fr10_leaf_steps` machinery (:11031-11045) generalizes it. Each NEW shape = ~15-30 lines: one exact-match
guard (`_fr10_is_<name>`, like `_fr10_is_cat3w` :11026) + a `torch.stack` packing in `(len,path)` order
(like cat3w :11515-11538); default cat9 untouched; the disengagement RAISE (:12005) intact. The launcher
auto-derives `num_speculative_tokens = len(TREE)`.

### R4 — `cat6root` : ROOT-HEAVY caterpillar (full spine, remove ALL off-root leaves, fan the root) [RANK 1]
- TREE (6 nodes, committed-spine depth 5, N=6 ⇒ **pad8**):
  `[(0,), (1,), (0,0), (0,0,0), (0,0,0,0), (0,0,0,0,0)]`
- = the FULL depth-5 spine (keeps cat9's deep d0-d4 spine accept) + ONE root sibling `(1,)`, **zero
  off-spine leaves**. `_fr10_spine_steps = 4`, `_fr10_leaf_steps = {}`, root sibling in slot 1 (reuse the
  cat3w `_fr10_root_leaf_token` capture verbatim).
- **WHY RANK 1:** isolates "does the d0 root rescue ALONE (no deep leaves) net-beat native on TPS at
  pad8?" Keeps the full SPINE accept (the deep spine, NOT the lossy deep leaves) + the +0.035 d0 rescue,
  at N=6 pad8 (sheds the entire pad8→pad16 step, the dominant +42-46 ms). The root sibling is strict-mask
  invisible to the spine (§2a-iii) ⇒ ~0 spine-perturbation flips — closest to chain5's lossless regime
  PLUS a d0 accept bump. The OFF-root leaves (cat9's 4 deep leaves) were the weak-accept AND lossy-
  co-residency part; dropping them is the cleanest fast+near-lossless move.

### R5 — `cat6root_g` : R4 root sibling CONFIDENCE-GATED [RANK 2 — leanest deploy form]
- Same 6 nodes; `(1,)` emitted only on near-tie root. draft-toks/event ∈ {5,6}. Full spine (pad8) + a
  free, sparse d0 rescue. Likely the best TPS/lossless trade if R4's accept clears break-even.

### R1 — `cat7rd` : REMOVE-DEEP (drop the d4 spine + d5 leaves to pad8) + ADD-ROOT sibling + 3 leaves [RANK 3]
- TREE (7 nodes, committed depth 4, N=7 ⇒ **pad8**):
  `[(0,), (1,), (0,0), (0,1), (0,0,0), (0,0,1), (0,0,0,0)]`
- = cat9 with the depth-5 spine node `(0,0,0,0,0)` and BOTH depth-5 leaves removed (remove-deep) + a root
  sibling `(1,)` added (add-root). Spine depth 4; leaves at d1,d2,d3; root sibling at d0.
  `_fr10_spine_steps = 3`, `_fr10_leaf_steps = {1,2,3}`, root sibling slot 1.
- **WHY:** N=7 is the LARGEST tree that stays in pad8 (next_pow2(8)=8) — sheds the pad8→pad16 step while
  keeping 4 spine depths (cat9 d0-d3 rates 0.871/0.828/0.638/0.483) + 3 leaves + the root rescue. More
  accept than R4 (3 extra leaves) at the same pad8, BUT re-introduces 3 off-root leaves = SOME
  co-residency (the cat3w lesson: shallow width is not free). The accept-vs-co-residency knee.

### R2 — `cat7rd_g` : R1 with the root sibling CONFIDENCE-GATED [RANK 4 — R1's deploy form]
- Same 7 nodes; `(1,)` gated on near-tie. draft-toks/event ∈ {6,7}. On confident roots serves the 6-node
  `[(0,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0)]` (still pad8). Free top2-margin gate.

### R3 — `cat3w` (ALREADY ON HEAD, no code) : aggressive remove-deep, depth-3 root-heavy [RANK 5 — pad8 floor probe]
- TREE (5 nodes, committed depth 3, **pad8**): `[(0,), (1,), (0,0), (0,1), (0,0,0)]`. Built & exact-match
  guarded on HEAD (`_fr10_cat3w_choices` :11005). Use as the pad8-floor SPEED probe (cheapest accept-
  bearing tree) + depth-3 accept reference. NOTE its flips were ~17 (LOSSY) — it is a speed/depth
  reference, NOT a lossless candidate; pair its s/fwd with R1's to size the d4-removal speed delta within
  pad8.

**L3 conf-gated ROOT-SIBLING EMIT (the wgb0yegin lever; applies to R2/R5/cat10-gated):** emit `(1,)`
**when the root top-1 margin < tau** (62% of rejects are step-0 — FR13_MATH_HISTORY:151; the root
runner-up is truth 27% on those near-ties). Implemented via the existing
`_fr10_top2 = torch.topk(_fr10_logits, 2)` (patcher :11167-11181, runner-up already captured for cat3w) +
one `(top1_logit − top2_logit) < tau` scalar that conditionally appends slot 1. FREE (no extra lm-head).
The committer/verify/mask auto-adapt off the per-event `tree_choices`/parent-indices.

**Realizability caveat (class 9, FR13_RESHAPE_SHAPE_DESIGN §4):** all R1-R5 use **rank-2 width ONLY**
(root runner-up + spine runner-ups, all captured by `torch.topk(...,2)`). NO rank-3 node exists in the
drafter (only `[:,1]` captured) — do NOT propose 3-way fans. Each new shape FAILS LOUD via the
disengagement RAISE until its packing branch + exact-match guard is added (hand to the GPU worker;
behavior-preserving, flag-gated, default cat9; NEVER `FR10_ALLOW_LINEAR_FALLBACK`).

---

## 4. ACCOUNTING-CORRECT MEASUREMENT (AVOIDS the cat10 [2,6,8,6] artifact)

The cat10 mis-read came from **cross-trajectory whole-window accept/event + the sibling-stop denominator**.
This protocol removes both. Reuse `scripts/fr13_shape_gate.sh <name> "<TREE>"` (serialized GPU,
recover_host_memory + MemAvailable≥95GiB + docker-empty before each boot, locked pipeline flags).

1. **ENGAGEMENT (class 9) FIRST — before any number.** Assert `tok/draft == len(TREE)` (ungated) or
   `tok/draft ∈ {realizable set}` (gated, e.g. {6,7}), NOT a fixed int for gated shapes. Also assert
   `has_tree_parent_indices` + `tree_sample_accept`. The patcher RAISES "caterpillar drafter disengaged"
   for any unbuilt shape — a vacuous tree FAILS LOUD, records nothing (fail_loud_assert_engagement).
2. **WITHIN-BOOT DETERMINISM (class 8):** rep1≡rep2 byte-identical served streams, all prompts, greedy
   AND t0.6. The cross-boot byte gate is BANNED (feedback_no_cross_boot_byte_gate).
3. **LOSSLESS (held per-change, depth-AGNOSTIC):** per-token clear-margin argmax flips vs **THIS boot's
   OWN no-spec RECURRENT decode oracle** (`FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN, teacher-forced
   max_tokens=1 per served position on the byte-identical served prefix), thr 1.0 nat
   (`fr13_oracle_stream_teacher_force.py` / `fr13_gold_margin_probe.py`, the confound-free instrument).
   Compare = US vs native-E5, **each vs its OWN no-spec recurrent oracle**, NEVER chunked-prefill /
   streamed-logprobs / serial-torch / a backend NAME (int-view never atol;
   feedback_fr13_lossless_compare_target). Assert `spec_metrics_delta_during_oracle == 0`. Branch B's gate
   = "flips stay at the cat9 operating point (HELD, not regressed)" — it need NOT improve flips.
4. **ACCEPT — DEPTH-MATCHED + PAIRED teacher-forced (the artifact-avoider):**
   - **DEPTH-MATCH:** committed-depth-5 shapes (cat10, R4/R5) → **native E5**; committed-depth-4 (R1/R2)
     → **native E4**; depth-3 (R3) → **native E3** — and **E3/E4 are UNMEASURED on the
     temp0/prompts_swe4/recurrent frame; CAPTURE them before judging any d≤4 arm "slow"**
     (feedback_depth_matched_accept_compare; NEVER a d4 arm vs E5).
   - **PAIRED teacher-forced, per-event, NOT cross-trajectory whole-window:** score accept on a
     byte-identical served prefix shared with the depth-matched native; the aggregate cross-trajectory
     accept/event is class-12 confounded (the cat10 trap). Report PER-DEPTH accept RATE (d0..d4) + the
     within-arm **d0-rescue delta**, not whole-window accept/event.
   - **De-confound the sibling-stop denominator:** with FR10_METRICS=1 + a per-node sibling-vs-spine d0
     tag, REMOVE sibling-win (`accepted_len=1` root) events from the `per_pos[0]` denominator before
     computing d1|d0. This is the per-node counter the prior cat10 boots LACKED.
5. **SPEED — `decode_seconds` RAW counter ONLY** (`vllm:request_decode_time_seconds_sum /
   vllm:spec_decode_num_drafts_total`), metrics OFF, per-request/per-event, BI=0 pinned IDENTICAL both
   arms, prompts_swe4 seed 1313 greedy temp 0.0. NEVER TPS/accept-decomposition as a measured fact, NEVER
   wall (reference_fr10_speed_measurement_pitfalls; feedback_dont_handroll_speed). Record `verify rows` +
   `n_pad` per arm to attribute the pad8-vs-pad16 step. decode_seconds/draft + paired per-depth accept are
   the ONLY load-bearing numbers; TPS is DERIVED for reporting, not gated on.
6. **VERDICT instrument:** TPS = (paired accepted tok/event) / (decode_seconds per fwd), arm vs **native
   E5** (the GOAL bar = cat9-family strictly > native E5 at B=1). WIN iff: lossless HELD (flips not
   regressed vs the cat9 operating point, each-vs-own-oracle) AND decode-TPS > native E5 with accept ≥ the
   depth-matched break-even. Per-layer 0.0 = DEV check only; within-floor e2e is the gate.

**Sweep order (GPU serialized, AFTER prelaunch host-mem protocol):** R4/`cat6root` (full-spine pad8 floor)
→ R1/`cat7rd` (pad8 + 3 leaves) → cat10 UNGATED (the artifact-corrected re-measure, per-node counter ON)
→ then the gated forms (R5/R2/cat10-gated) for the deploy d0-rescue. CAPTURE native E3/E4 references in the
same campaign (UNMEASURED today). chain5/cat9 only for a fresh same-boot oracle re-baseline.

---

## 5. PER-CANDIDATE SPEED + ACCEPT + LOSSLESS PREDICTION (depth-matched, INFERRED-labeled)

s/fwd basis: chain5 pad8 ≈ 0.222 post-FIX-3 (0.3517 legacy); cat9 pad16 ≈ +42-46 ms/fwd over chain5's
pad8 (the dominant step); each verify ROW ≈ +0.0019 ms (M-invariant, free). All ms INFERRED until the
clean boot. Flips each-vs-OWN-recurrent-oracle (depth-agnostic); accept depth-MATCHED + PAIRED.

| cand | shape | N / pad | verify rows | committed depth | predicted s/fwd (INFERRED) | predicted accept (depth-matched, PAIRED) | predicted lossless (HELD) | net TPS vs native E5 |
|---|---|---|---:|---:|---|---|---|---|
| **R4 cat6root** | spine5 + root sib | 6 / **pad8** | 6 | 5 | **~chain5 pad8** (sheds pad16 step, −40+ ms vs cat9) | full-spine accept (cat9 d0-d4) + d0 rescue +0.035; root sib strict-invisible; vs **native E5** | **best**: ~chain5 regime (1-5 flips) + strict-invisible sparse sib (~+0 spine perturb) | **most likely > native** (pad8 s/fwd, near-cat9 accept) |
| R5 cat6root_g | R4, root gated | 6→5 / pad8 | 5-6 | 5 | ≤ R4 (fewer rows on confident roots, free) | R4 accept on near-ties only, no dilution | ≥ R4 (sparser sib) | ≥ R4 |
| R1 cat7rd | spine4 + root + 3 leaves | 7 / **pad8** | 7 | 4 | ~chain5 pad8 + 1 row (free) ≈ pad8 | d0-d3 (cat9 0.871/0.828/0.638/0.483) + 3 leaves + d0 rescue; vs **native E4 (UNMEASURED)** | held @ cat9-operating-point; 3 shallower leaves = some co-residency (< cat9's 4 deep) | likely > native (pad8 + more leaves than R4) |
| R2 cat7rd_g | R1, root gated | 7→6 / pad8 | 6-7 | 4 | ≤ R1 | R1 accept, gated d0 rescue; vs **native E4 (UNMEASURED)** | ≥ R1 | ≥ R1 (deploy form) |
| R3 cat3w (HEAD) | spine3 + root + d1 | 5 / **pad8** | 5 | 3 | ~chain5 pad8 (lowest rows) | LOW (d0-d2 only); vs **native E3 (UNMEASURED)**; chain3 was 2.27 | LOSSY (~17 flips measured) — SPEED ref only, not a lossless cand | speed-floor probe; accept too low to beat native alone |
| cat10 (ungated) | cat9 + root sib | 10 / pad16 | 10 | 5 | ~cat9 (+1 row free; SAME pad16) ≈ +2.9 ms vs cat9 | cat9 accept + d0 rescue +0.035 (PAIRED, sibling-stop-de-confounded — NOT the artifact 2.932); vs **native E5** | held = cat9's 22 (FLAT; root sib redistributes, strict-invisible) | ~cat9 (no pad win); the d0 rescue is the only gain |
| cat10-gated | cat9 + gated root | 10→9 / pad16 | 9-10 | 5 | ≤ cat10, ~cat9 | cat9 + sparse d0 rescue, no dilution | held = 22 | ~cat9 + a hair |

**Reading:**
- **The pad8 shapes (R1/R2/R4/R5) are the real TPS lever** — they shed the dominant pad8→pad16 step
  (~+42-46 ms/fwd) that cat9/cat10 pay. cat10 (still pad16) does NOT win on s/fwd; its only gain is the
  +0.035 d0 rescue, so cat10's value is **as the d0-rescue building block bolted onto a pad8 shape**
  (R5/R2), not standalone.
- **R4/cat6root is the lead candidate:** pad8 s/fwd (near chain5) + the full depth-5 SPINE accept (the
  deep spine, NOT the lossy deep leaves) + a strict-mask-invisible (≈lossless) root sibling — the closest
  thing to "chain5's lossless speed regime + cat9's spine accept + a free d0 bump." OPEN risk: whether the
  full spine WITHOUT the 4 deep leaves holds accept ≥ break-even (the deep leaves carried SOME accept; the
  paired E5 capture decides).
- **Lossless is HELD, not improved, by Branch B.** The removed deep leaves slightly help the co-residency
  carrier, but the goal here is speed-with-lossless-held; the 22-flip chase is the parallel drift track.
- **Break-even:** for a pad8 shape to beat native E5 on TPS it needs accept ≳ (s_pad8 / s_native) ×
  native_accept ≈ (0.222 / 0.218) × 3.08 ≈ 3.14 — i.e. roughly cat9's accept. R4's full spine + d0 rescue
  is the candidate most likely to hold that; the paired capture is decisive.

---

## Cross-refs
`FR13_RESHAPE_AB_RECURRENT_BIND.md` (depth+1/width+16 carrier, recurrent frame canonical),
`FR13_RESHAPE_EXHAUSTED_BIND.md` (reshape can't do BOTH lossless+fast on flips → Branch B = speed-held),
`FR13_RESHAPE_SHAPE_DESIGN.md` (infra/feasibility, rank-2-only, Sequoia/TALON),
`FR13_CAT10_BIND.md`, `FR13_CAT10_INVESTIGATE_BIND.md` (the artifact decomposition + confidence-gate lever),
`FR13_SPEED_HISTORY_RECONCILE.md` (M-invariant lm-head, N_PAD step, FIX-1/2/3),
`FR13_SPEED_TAX_BASELINE.md` (per-node +42-46 ms, pad8/pad16, 3N+2a+1 floor),
`FR13_MATH_HISTORY_RECONCILE.md` (62%-step-0, conf-gated root), `FR13_SPEED_TUNING_PLAN.md` (Branch A),
`scripts/fr13_shape_gate.sh`, `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (:10984-11045 choices+leaf_steps,
:11155-11181 root capture, :11505-11538 cat3w packing, :12005 disengage RAISE),
[[project_fr13_tree_reshape_unifying_lever]], [[feedback_check_artifact_before_concluding]],
[[feedback_depth_matched_accept_compare]], [[reference_fr10_speed_measurement_pitfalls]],
[[feedback_dont_handroll_speed_defer_tuning]], [[feedback_fr13_lossless_compare_target]],
[[reference_scalar_metric_per_token_blindspot]], [[feedback_wy_parked_dont_revive]] (WY out of scope).
