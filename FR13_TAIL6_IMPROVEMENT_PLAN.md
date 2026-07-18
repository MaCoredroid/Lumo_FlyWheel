# FR13 tail6 improvement plan — two compounding directions

**Baseline (locked, same 16-task B=4 SWE-Verified, temp 0.6):**
- Deliverable **tail6** (MTP spine d1–5 + 2 branches/depth + arctic tail d6–11, n_pad=32/BV=8): **accept ≈ 5.2**, derived_tps_gpu ≈ **70**, lossless never-regress.
- Depth-5 baseline (t33333): accept 3.697, tps_gpu 67.06.
- Pre-warm proven ≈0 same-session (dropped). The **arctic tail alone** carries +42% accept.

Two independent levers, and they **multiply**: cheaper per-step (Dir-1) turns accept into more TPS; higher accept (Dir-2) amortizes the per-step cost. TPS ≈ committed_per_step / (draft+verify+commit ms).

---

## Grounded data

### A. Per-depth acceptance profile (656-window aggregate, tail6)
| depth | zone | survival P(reach d) | conditional P(hit d \| reached) |
|---|---|---|---|
| 1 | head/spine | 0.970 | 0.970 |
| 2 | head | 0.838 | 0.864 |
| 3 | head | 0.713 | 0.851 |
| 4 | head | 0.605 | 0.849 |
| 5 | head | 0.524 | 0.866 |
| 6 | tail (MTP→arctic handoff) | 0.349 | **0.666** |
| 7 | tail | 0.296 | 0.848 |
| 8 | tail | 0.265 | 0.895 |
| 9 | tail | 0.240 | 0.906 |
| 10 | tail | 0.218 | 0.908 |
| 11 | tail | 0.207 | 0.950 |

Σ survival = **5.23 = accept**. Two structural facts:
- **Head conditional is flat ~0.85** (d1–5); the ~15%/depth loss is MTP top-1 misses the 2 branches don't rescue. Branch-accept probe (`fr13_analyze_branch_topp.py`): argmax falls **spine 84.5% / branch 10.2% / hard-miss 5.3%**.
- **The tail's weakest link is d6 = the MTP→arctic handoff (conditional 0.666)**; deep tail d7–11 actually *holds* at 0.85–0.95 (repetitive spans). So the tail loses most of its potential *at the first arctic token*, then coasts.

**Loss decomposition** (vs a hypothetical all-accept = 11): accept 5.23; **head attrition ≈ 1.35**, **tail attrition ≈ 1.57** (of which ~0.43 is the single d6 handoff drop from 0.866→0.666).

### B. Per-stage GPU cost (per decode STEP, B=4)
| stage | ms/step | HW floor | nature | reclaimable |
|---|---|---|---|---|
| DRAFTER | ~100 | ~22–30 | 4 sequential M=1 MTP-head forwards (FIXED vs tree size) + host arctic trie walk + assembly/H2D | ~70 (launch-latency + host, NOT bandwidth) |
| VERIFY | ~170 (B=1) / ~340 (B=4) | 98.6 (27 GB weight read) | the one full-model tree forward; the ONLY genuinely GPU-bandwidth-bound stage | ~70 (M=25 GEMM eff + GDN serial scan; the O(N²) gather is NOT it — parent-gather was −1.7%) |
| COMMITTER | ~74–113 | ~sub-ms compute | GDN replay is sub-ms/12k-CTA; the rest is a DtoH-sync bubble + host orchestration (48-layer `.item()` storm) | ~80 (host/sync, but much is B=1 serial latency) |

Key: **drafter GPU is FIXED in tree size** (MTP head always drafts 5 depths; branches are ~free `topk` reads; tail adds only HOST time). So growing the tree costs **verify + committer**, not drafter GPU — and higher accept amortizes all three.

---

## DIRECTION 1 — Make tail6 cheap (drafter + verify + committer)

Goal: cut ms/step so accept 5.2 buys more TPS. Every lever must stay **byte-identical / never-regress** and be flag-gated with a same-boot A/B.

Ranked by reclaim / risk (from workflow wbvlbn0x3, source-verified):

1. **LEVER 1 — dead metrics-dict elim (DONE, committed).** ~1 ms, byte-identical.
2. **LEVER 2 — route GDN replay to the batched `_ep_launch_all` (kill the 48-layer `.item()` storm).** ~5–10 ms, LOW-MED risk, **localized + designed** (behind `FR13_REPLAY_BATCHED_RUNROW`; under `_fr13_runrow_commit=True` the per-layer publish is already dead → batched path is byte-identical). *Next to ship.*
3. **LEVER 3 — kill the 7 per-launch `.contiguous()` copies** (patcher :5142–5148, 48×7=336 D2D/step). LOW effort.
4. **LEVER 4 — repoint the next-step ATTN-KV remap at the device buffer** (drop a HtoD round-trip). ~0.5–2 ms.
5. **LEVER 5 — whole-spine drafter CUDA-graph capture. ~40–60 ms, the only LARGE reclaim, HIGH risk.** The MTP spine has **no `.item()`/sync** (grepped) and static shapes → capturable in principle; 4 sequential M=1 launches collapse to one replay. 5 hard invariants (N_PAD-inv, M-inv of `in_proj_ba` = the SLOT_REORDER problem, per-level bit-exactness, no-sync-in-capture, req-key routing survives).
6. **VERIFY GDN-scan occupancy (open):** verify is 1.7× its 98.6 ms floor; the residual is M=25 GEMM efficiency + the GDN serial recurrence at ~1 CTA/SM (128 KB h_cache pins occupancy). SRAM-staging h_cache (LEVER-2 of the kernel study) frees registers/de-spills but does NOT raise occupancy past ~1 CTA/SM losslessly. Needs an ncu pass to size.

**Plan (cheapest-first, each gated byte-identical then measured):** L2 → L3 → L4 (bank ~7–14 ms cheaply), then attempt L5 (the big one) behind its flag with the same-boot bit-exact self-check. Expected: ~15–70 ms/step off ~550 → meaningful TPS with zero accept change.

**Honest caveat:** at agentic effective-batch ≈1.3, cross-request overlap is unavailable, so much of the committer's span is fundamental B=1 serial latency (committed token needed before the next drafter forward). L5 is the only lever that moves the needle a lot, and it's the riskiest.

---

## DIRECTION 2 — Save misses (raise accept)

The committer is **monotone `accept=p(S)`** → *adding* candidates can only raise accept (never-regress, Gate1 32/32). So every miss-recovery lever is lossless by construction; only magnitude and cost are open. Ranked by expected accept / cost:

1. **d6 handoff repair — the single biggest lever (conditional 0.666).** The MTP→arctic transition loses 33% at the first tail token, and every deeper survival is scaled by it (compounds). Levers:
   - **Add branches at d6** (currently the tail is spine-only): give d6 the same 2–3 candidates the head gets (MTP-guided suffix ∪ arctic top-k). If d6 conditional 0.666→~0.85, tail sum ≈1.575→~2.0 → **+0.4 accept**. Cost: +2 nodes at n_pad, some verify.
   - **Better d6 seed:** the arctic walk is seeded by `_COMMITTED`; seed it with the MTP-head's depth-5 token so the first tail token is MTP-anchored, not a cold arctic guess.
2. **Suffix COMPLEMENT branches on the head (d1–5) — raise the flat 0.85 conditional.** Add an arctic/suffix candidate **alongside** MTP's top-3 at each head depth (fills part of the 5.3% hard-miss). Arithmetic: conditional 0.83 → 0.83 + 0.17·f, f = fraction of misses the suffix covers. Ladder-step-1 says head misses are only ~5.3% (wider tree = limited upside) → **+0.1–0.3 accept**, cheap (branches are ~free `topk` reads on the drafter; cost is verify n_pad).
3. **Tail branches at d7–11 (secondary).** Deep tail already holds 0.85–0.95 conditional, so branches there catch little and cost n_pad — do only if the 32-node budget allows after (1).
4. **Deeper tail (d12+) — windfall only.** Tail conditional rises with depth on repetitive spans, so extending past d11 pays off *only* on long exact repeats; costs verify depth + risks capture stalls (depth-21 hung). Gate on the repetitive fraction, don't extend blindly.

**Node budget:** n_pad=32 with BV=8 (byte-identical register budget). Current tail6 uses 21 nodes → **11 free slots**. Spend them on d6 branches (biggest ROI) first, then head-complement.

**Plan:** (a) d6 branches + MTP-anchored seed → target tail sum +0.4 (accept ~5.6); (b) head suffix-complement → +0.1–0.3; both within the 32-node horizon, both lossless. Gate: live B=4 A/B, accept UP, per-depth survival re-measured, garble-clean.

---

## Combined roadmap (both, compounding)

| phase | direction | change | expected | risk |
|---|---|---|---|---|
| P1 | speed | LEVER 2 batched replay (ship behind flag) | −5–10 ms/step | LOW-MED |
| P2 | accept | d6 branches + MTP-anchored seed | +0.4 accept | LOW (monotone) |
| P3 | speed | LEVER 3+4 (contiguous copies, KV-remap device buf) | −5–10 ms | LOW |
| P4 | accept | head suffix-complement branches | +0.1–0.3 accept | LOW (monotone) |
| P5 | speed | LEVER 5 drafter CUDA-graph (the big one) | −40–60 ms | HIGH |

Net target: **accept ~5.6–5.8 at meaningfully lower ms/step** → TPS well above baseline, still lossless. Every accept lever is never-regress by construction; every speed lever is byte-identical-gated. Measure each on the live B=4 gate before the next (design §9 discipline).

**Why both:** TPS = committed / step-ms. Dir-2 raises the numerator (5.2→~5.7), Dir-1 cuts the denominator (~550→~480 ms). Multiplied, that's a ~30%+ TPS gain on top of a lossless accept improvement — the accept and speed wins reinforce rather than trade off.

---

## Direction-2 IMPL + the two-proposer sweep (2026-07-16)

**Framing (user):** the tree is a MERGE of TWO tree-proposers, each proposing top-k per depth —
- **MTP** (strong, ≤5 heads) covers head depths 1..**n**; **x=0** ⇒ pure MTP (= cat33333).
- **Suffix/Arctic** (weaker/token, unbounded depth, catches MTP misses) covers the **x** tail depths past the head, and *optionally* complements the head (overlap); **n=0** ⇒ pure arctic.
Both propose **trees** — we currently throw the arctic tree away and use only its top-1 chain in the tail.

**IMPL done (CPU-tested, no config drift):** `tail_tree_order` + `assemble_tail_tree` gained `tail_branches`/`tail_branch_depths` (default 0 ⇒ byte-identical shipped tail6). Branched tail fills the d6/d7 siblings from the **arctic tree runner-ups** `suffix_rel[j][1:]` (already computed by `arctic_tree_to_suffix_rel`, just discarded today). Pad-with-spine ⇒ monotone-lossless. TAIL6B tree string (25 nodes = 15 head + 6 tail spine + 4 d6/d7 branches) generated.

**Measured per-depth conditionals (656-window tail6):** head d1–5 ≈ 0.85 flat; tail j0=**0.666 (the MTP→arctic handoff = weakest link)**, j1–5 = 0.848/0.895/0.906/0.908/0.950. The handoff is where the arctic is still ambiguous — exactly where runner-up branches help.

**Sweep tool `fr13_tail_config_sweep.py`** models accept = Σ survival over (n, x, w_over=arctic-complement, w_tail/tail_bd=tail branches) within n_pad=32, with TWO unmeasured uplifts:
- `tail_uplift` — how much an arctic branch raises a tail depth's conditional (d6-handoff rescue).
- `comp_uplift` — how much an arctic complement raises an MTP depth's conditional (head-miss rescue).
Calibrated at tail6 (n5,x6,no-branch → predicts 5.23 ≈ measured 5.2). **CAVEAT (honest):** the tail-conditional plateau (~0.95) is fit to tail6's MTP-anchored tail on THIS workload; the model's extreme optima (n=1,x=21) are **extrapolation artifacts** — Front-2 already showed arctic-heavy configs lose. Trust the model for interpolation near tail6, not for the extremes.

**Calibration-first experiment ladder (each on the SAME subset_b4_sixteen, no config drift — only the tree/flag differs):**
1. **A/B-1 (this iteration): tail6 (n5,x6, spine tail) vs TAIL6B (n5,x6, w_tail=2,tail_bd=2).** Same n,x — isolates `tail_uplift` (the d6/d7-branch effect). Expected +0.2–0.4 accept if the handoff rescue is real.
2. **A/B-2: tail6 (x6) vs deeper tail (x10, spine).** Calibrates whether the tail conditional actually holds past d11 (does extending x pay, or decay?).
3. **A/B-3: head-complement (w_over=1) vs tail6.** Calibrates `comp_uplift`.
Then re-run the sweep with the 3 calibrated numbers → pick the accept-max config within budget → final live gate.

**Endpoints as data points:** x=0 (pure MTP, accept 3.65) and n=0 (pure arctic) bound the space; both are cheap sanity A/Bs if the interpolation looks off.

## Direction-2 EXPERIMENT RESULT: tail-DEPTH (x10) = NO (2026-07-16)
tailx10 (25-node, depth-15 spine tail, SAME config as tail6) canonical (4 tasks): **accept 4.876, tps_gpu 55.0**
vs tail6 ~5.1 / ~70. accept NOT up (arctic is cold past d11 -> deep tail = pad = never-regress floor) AND tps
DROPS ~21% (deeper 25-node tree = more verify + GDN-scan). => extending the spine tail is NET-NEGATIVE; the
sweep's 0.95 deep-tail plateau was an EXTRAPOLATION ARTIFACT (the measured d7-11 rise was the limited depth-11
window). x-axis lever DEAD. Pivot fully to the d6-BRANCH lever: raise the handoff conditional (0.666) WITHOUT
adding depth cost (branches are within the existing depth, +2 nodes at n_pad, ~free drafter topk).

---

## Direction-2 d6-branch: clean same-session A/B LAUNCHED (b7)

**What ran before:** a SOLO `tail6b` run (b6) decoded clean — `TAIL[fired=4619 hit=4501 cold=8]`,
zero crashes — proving the branched wiring works live. But (a) its needle predates the `br_real`
counter so branch-vs-pad engagement was unproven, and (b) its accept vs *cross-session* tail6 (~5.1)
is confounded. Killed it.

**The honest measurement (running now):** `scripts/fr13_tail6b_ab_seq.sh` via the b4 campaign driver
on `subset_b4_sixteen`, `RUNROOT=output/fr13_tail6b_ab TAG=b7`, back-to-back arms in ONE driver run:
1. `tail6b` (25-node, d6/d7 arctic-branched, depth-11) — the deliverable lift
2. `tail6`  (21-node spine tail, depth-11) — the never-regress bar

The **ONLY** difference is `FR13_TAIL_BRANCHES=2 / FR13_TAIL_BRANCH_DEPTHS=2` (+ the 4 branch nodes in
the tree). GPU_UTIL=0.72 / geom BV=8 / tail-mode / draft-source=merged / no-prewarm are IDENTICAL on
both arms ⇒ **no config drift**. WALL=1800, B=4, CONC=4 (== the deployment gate).

**Reads to take (next fires):**
- `br_real > 0` in the tail6b `[FR13_MERGED ENGAGED]` needle ⇒ branches carry REAL arctic runner-ups
  (not pad-fallback). This is the engagement gate; if br_real==0 the branched path is vacuous.
- `deploy_speed accept_per_event`: tail6b vs same-session tail6. Monotone-lossless ⇒ tail6b ≥ tail6 on
  the same spans; the delta = the d6/d7 handoff lift (targets the measured weakest link, cond 0.666).
  Target ~5.5 at unchanged tps (same depth-11 ⇒ tps ≈ tail6, unlike tailx10's depth-15 −21%).
- If tail6b > tail6: the head-complement lever (arctic ∪ MTP branches at d1–5) is next.
- If tail6b ≈ tail6 with br_real>0: branches engage but don't help (handoff isn't the accept leak) —
  re-target the lever.

### b7 update: ENGAGEMENT GATE PASSES + diagnosis + next lever wired

**Diagnosis (airtight, CPU-confirmed):** arctic's first tail token (d6) IS re-anchored to the
committed MTP head — `decide_tail` builds `pattern = committed + head` (the d1-5 MTP spine) and feeds
that to `cache.speculate` (fr13_merged_drafter.py:384). So the measured 0.666 handoff conditional is
**inherent to arctic's first post-handoff prediction, NOT a re-anchoring bug.** The lever is therefore
seam candidate *coverage*, exactly what tail6b tests. Arctic's runner-up branches come from its suffix
**tree siblings** at depth d, ranked by prob (arctic_suffix_adapter:131-144) — same suffix-match context,
so potentially correlated; `br_real` gates whether they exist at all.

**LIVE engagement (b7, tail6b arm serving):** `TAIL[fired=509 hit=484 cold=0 br_real=466]`. br_real≈fired
⇒ nearly every tail speculation yields REAL arctic runner-up branch tokens (not pad). The branched path
is non-vacuous; arctic's tree HAS siblings at the seam. Engagement gate ✓. Open: does it raise accept
(deploy_speed, on run completion).

**Next lever wired + CPU-tested (ready, not yet run — b7 owns the GPU):**
- `tail6c` KIND: seam-CONCENTRATE — all 4 branches at d6 ONLY (BRANCHES=4 DEPTHS=1), 25 nodes / n_pad=32
  / tps == tail6b, vs tail6b's 2+2 spread across d6/d7. Tests concentrate-vs-spread at THE seam (d6 leak
  0.334 >> d7 0.152). Needed `build_tail_branch_columns` width = max(3, tail_branches+1) (was hardcoded 3,
  would drop ranks 3,4); no-op for BRANCHES<=2 ⇒ tail6b + running b7 byte-identical.
- Decision tree keyed on b7's accept delta:
  - tail6b > tail6 (br_real>0 ✓ already): arctic seam branches help ⇒ run tail6c (concentrate) vs tail6b.
  - tail6b ≈ tail6 despite br_real>0: siblings are correlated (same-match) ⇒ escalate to a DECORRELATED
    seam candidate = an MTP-extended d6 token (real model head vs suffix match) injected as a d6 branch.

**Leverage math (why d6):** accept gain from raising cond_d by δ ≈ δ · survival_{d-1} · Σ(tail survivals
from d). d6: survival_5=0.524, raising cond_6 0.666→0.766 (δ=0.10) scales survivals d6-11 (Σ=1.575) by
1.150 ⇒ +0.236 accept. Biggest single-depth lever; head-complement at d2-5 gives ~+0.15 each but those
conditionals (0.85-0.97) are already near-saturated (MTP branches present) ⇒ secondary.

### Calibration bars for b7 (validate accept_per_event against these)

Per-depth conditional-acceptance model (calibrated: tail6 baseline = 5.227, matches measured 5.1-5.24).
Branch at depth d recovers fraction r of that depth's miss (1-cond_d); r encodes sibling decorrelation.

| arm    | change                    | r=corr(.15) | r=decorr(.30/.45) |
|--------|---------------------------|-------------|-------------------|
| tail6  | spine tail (baseline)     | 5.227       | 5.227             |
| tail6b | 2+2 branches @ d6,d7       | 5.381       | 5.540             |
| tail6c | 4 branches @ d6 only       | 5.464       | 5.583             |

**Gate read (b7 tail6b accept_per_event, depth-matched vs native MTP-5 E5, tps must stay ~= tail6):**
- tail6b in [5.38, 5.54] => arctic seam branches help (decorrelation 15-30%) => run tail6c (concentrate) next.
- tail6b < 5.35 => branches barely help despite br_real>0 (siblings correlated, same suffix-match) =>
  escalate to a DECORRELATED seam candidate: an MTP-extended d6 token (+1 MTP forward, drafter:13211
  cap 4->allow one extra head step) injected as a d6 branch. Real model head vs suffix match => decorrelated.
- tail6b ~= 5.23 => no help => same escalation, higher priority.

Note the MTP-d6-seam lever is NOT free (+1 MTP forward ~ +1/5 head drafter cost); it's a speed-vs-accept
tradeoff, justified only if arctic width plateaus below the decorrelated bar. tail6c (free, same tps) is
tried first. Live b7 engagement stable: TAIL[fired=2668 hit=2553 cold=0 br_real=2044], 0 crashes.

### b7 INTERIM (caveated, NOT a conclusion)

tail6b arm 4/16 tasks: token-weighted mean accept over 217 windows / 17482 accepted tokens = **5.418**.
Squarely in the calibration "branches help" band [5.38, 5.54] (recovers ~15-30% of the d6 miss). CAVEATS:
(1) 4/16 tasks (interim); (2) window-mean token-weighting APPROXIMATES the true accept_per_event (the
deploy_speed reducer does it rigorously from /metrics brackets at arm end); (3) vs CROSS-session tail6
(~5.23) = the exact confound this A/B exists to remove -- the clean read is the same-session tail6 arm.
So: ENCOURAGING + consistent with branches helping, but NOT concluded. Await tail6b + tail6 arm finals.

Next experiment prepped (fr13_tail6c_ab_seq.sh, ready when b7 frees the GPU IF tail6b > tail6):
tail6c (concentrate 4@d6) vs tail6b (spread 2+2) same-session -- both 25 nodes/n_pad=32/same tps, only
branch distribution differs (no drift). Isolates concentrate-vs-spread at the seam.

**tail6c PRE-HARDENED (CPU end-to-end):** tail_tree_order(tb=4,tbd=1)=25 nodes; seam pp=5 needs width>4;
both real fill (wide[5]=width 5) AND cold-forward pad-fallback (max(rk+1,3)=5) satisfy rk<width for all
parent_pos. The width/rank-mismatch crash that hit tail6b 3x is pre-cleared => tail6c boots clean first try.

### b7 interim rising (5.589 @ 454 windows) + seam-geometry sweep ready

tail6b interim token-weighted accept = **5.589** (37784 tokens, 454 windows) -- risen from 5.418 @217w,
now at/above the calibration decorrelated bar (5.54). Backing out the model: tail6b≈5.59 => arctic siblings
recover ~35% of the d6/d7 miss (strong decorrelation, branches clearly help). SAME caveats (cross-session,
window-approx) -- clean read = same-session tail6 arm. br_real=8170, 0 crashes.

**Seam-geometry sweep (all n_pad=32 / same tps / boot-hardened CPU end-to-end, monotone-lossless, no drift):**
| KIND   | branches           | nodes | fill width | status              |
|--------|--------------------|-------|------------|---------------------|
| tail6b | 2 @ d6,d7          | 25    | 3          | RUNNING (b7)        |
| tail6c | 4 @ d6 (concentr.) | 25    | 5          | prepped+hardened    |
| tail6e | 3 @ d6,d7 (wider)  | 27    | 4          | prepped+hardened    |

**Sweep plan (one GPU job at a time, each same-session vs tail6b as reference):** after b7's clean
tail6b>tail6 confirms branches help, run tail6e (scale-what-works, likely best given high recovery) and
tail6c (concentrate) to find the best seam geometry. If any plateaus below a decorrelated ceiling =>
the MTP-d6-seam lever (+1 forward) is the escalation. All three are free width plays first.

### CORRECTION: my interim "accept ~5.4-5.5" was a DILUTED mean_accept_length, NOT accept_per_event

Red-teamed my own interim numbers against real artifacts:
- Canonical tail6 deploy_speed: accept_per_event = **5.099**, committed_per_event = 6.099
  (output/fr13_tail6_prewarm/.../deploy_speed_pw16.json). accept_per_event = num_accepted/num_drafts
  (decode-only /metrics BRACKETS, prefill excluded) -- fr13_measure.py:602.
- vLLM's logged "Mean acceptance length" = **accept_per_event + 1** (verified: the 9.83 window had
  106 accepted / 12 events = 8.83 accept, +1 bonus = 9.83). So it == committed_per_event.
- My interim token-weighted "5.42/5.49/5.59" was raw-window mean_accept_length (== committed), AND it
  averages over prefill/tool-gap windows (accept~0) that DILUTE it downward. => it is NOT comparable to
  the accept_per_event calibration band [5.38, 5.54] (those are accept_per_event / Sigma-survival units).

**What survives (valid):** (1) ENGAGEMENT proven -- br_real=8170, branches carry real arctic tokens.
(2) MONOTONE-LOSSLESS guarantees tail6b >= tail6 by construction (the committer accepts a superset of
candidates) -- the SIGN of the effect is guaranteed; only the MAGNITUDE needs the clean number.
**What I must NOT do:** quote the raw-window average as "accept" (that is exactly the hand-rolled-speed
pitfall). The ONLY accept read is the bracketed deploy_speed accept_per_event at arm completion:
tail6b vs same-session tail6, both decode-only-bracketed, apples-to-apples. Await it; do not pre-conclude.

### b7 arm1 DONE: tail6b canonical accept_per_event = 4.500 (this regime)

Bracketed deploy_speed (output/fr13_tail6b_ab/tail6b_b7/deploy_speed_b7.json):
- **accept_per_event = 4.500**, committed_per_event = 5.500, derived_tps = 4.76, s_per_fwd_gpu = 0.105,
  prefill_frac = 0.453. 16/16 tasks.

RECONCILES with last fire's correction: raw-window committed ~5.49 minus 1 = 4.49 accept ≈ 4.500 canonical.
So the raw-window mean_accept_length WAS committed_per_event; -1 recovers accept (the correction was right).

REGIME NOTE: 4.500 is below the calibration band [5.38, 5.54] because that band was anchored to a stale
tail6=5.23 from a DIFFERENT regime. This SWE-agentic regime (prefill_frac 0.45, this 16-task mix) runs
lower -- the tail6-family artifacts span 4.28 (tailg4c) to 5.10 (prewarm). So the ABSOLUTE calibration
levels do NOT transfer across regimes; only the same-session DELTA (tail6b - tail6, monotone >= 0) is
valid. tail6b=4.500 sits above tail6_tailg4c=4.277 (cross-run, weak evidence). CLEAN delta pending the
same-session tail6 arm (arm2, now running ~2hr). tps gate: tail6b derived_tps 4.76 -- compare to tail6 arm.

### b7 arm2 (tail6 baseline) early + DRIFT CHECK PASS

- **DRIFT CHECK PASS:** tail6 arm br_real=**0** (vs tail6b 8170) -- the spine-only baseline emits ZERO
  branch tokens, and no /logs/fr13_tail_branches.cfg sidecar. The two arms differ ONLY by the branch
  flags. Clean A/B, no config drift.
- **Early same-basis signal (raw-window, same tasks/session/method):** tail6 committed ~5.102
  (accept ~4.10, 105 windows, FIRST WAVE only) vs tail6b committed ~5.49 (accept 4.500, FULL 16).
  Direction POSITIVE: tail6 < tail6b, consistent with branches helping + monotone bound (tail6 <= 4.500).
- CAVEAT: tail6 is PARTIAL (first wave) vs tail6b FULL -- the ~+0.4 magnitude is likely inflated by
  partial-vs-full task mix. The CLEAN delta = bracketed deploy_speed accept_per_event when tail6
  completes all 16 (~1.5hr). Do not quote +0.4 as the lift; only the sign is established so far.

### b7 arm2 converging: raw delta shrinking to ~+0.10 (as predicted)

tail6 raw-window accept rose 4.10 (105w) -> 4.402 (218w) as more tasks completed -- the partial-vs-full
inflation is shrinking, confirming last fire's caveat. Same-basis raw delta now: tail6b 4.500 - tail6
4.402 = **+0.10 accept** (modest, low end of calibration ~15% correlated siblings). Sign positive
(branches help + monotone), magnitude STILL settling (tail6 partial, 4/16 patches). CLEAN bracketed
delta at arm2 completion (~1.5hr). 0 crashes both arms.

Strategic read (pending clean number): if the delta is modest (~+0.10-0.15), arctic siblings are hitting
a CORRELATION ceiling (same suffix-match context) -> more arctic width (tail6c/tail6e) may plateau, and
the DECORRELATED MTP-d6-seam (+1 forward) becomes the higher-value lever. Decide AFTER the clean delta +
the tail6c/tail6e sweep -- do not pre-build the MTP-seam (risky patcher change) on a partial number.

### SPEED GATE surfaced: tree ~TIES native MTP-5 (cross-run) -> add native arm to the sweep

User asked how tail6b's tps compares to native MTP-5. Cross-run comparison (tail6b b7 vs native_nocache_qc4
= flash_ns5_nocache, closest regime pf 0.42/ec 1.86):

| metric (B=4)              | tail6b (TREE 25-node) | native MTP-5 | tree vs native |
|---------------------------|-----------------------|--------------|----------------|
| accept / forward          | 4.500                 | 3.336        | **+35%**       |
| per-stream decode tps     | 4.45                  | 4.60         | -3% (~tie)     |
| aggregate (throughput)tps | 9.67                  | 10.48        | -8%            |
| decode-kernel tps_gpu     | 52.4                  | 59.1         | -11%           |
| s / forward (GPU)         | 0.105s                | 0.073s       | **+43% slower**|

KEY: the tree's +35% accept is CANCELLED by its +43%-slower 25-node forward => ~TIE (even slightly behind)
on tps. At B=4 the extra tree-attention compute is real (nuances the "HBM-bound, accept is the only lever"
read). "4.76 tps" (derived_tps) is the concurrency-summed basis the json flags NOT-E5-comparable; ignore it.

CAVEAT: CROSS-RUN (different subset + b7 tree arms are cache-OFF matching native nocache, but different
campaign). b7 was tree-vs-tree, so NO clean same-session tree-vs-native number exists yet.

**Action:** fr13_tail6_speed_sweep_seq.sh -- 4-arm same-session campaign: nativemtp5(flash_ns5_nocache) +
tail6b + tail6e + tail6c, native+tail6b FIRST (speed-gate pair). Definitively answers: does ANY tree
geometry beat native MTP-5 on per-stream tps, or is the tree only an accept play? Launch when b7 frees GPU.
(Supersedes fr13_tail6_sweep3_seq.sh, which lacked the native bar.)

### b7 near-final: accept delta converged ~+0.18 (tail6b 4.500 vs tail6 ~4.32)

tail6 arm 12/16 (final wave = same hard tasks tail6b ran), raw accept settled to ~4.324 (575 windows)
after oscillating 4.10->4.40->4.32 with task mix. Same-basis delta: tail6b 4.500 - tail6 ~4.32 = **+0.18
accept** (raw; clean bracketed number at 16/16). Modest, real, lossless -- calibration low end (~correlated
arctic siblings). The tail6 deploy_speed (imminent) gives BOTH the clean accept delta AND the within-session
SPEED delta: does the +0.18 accept beat tail6b's +4-node (25 vs 21) slower forward? = the branch net-win test.

NEXT LAUNCH (when b7 done + GPU free + cleanup): the 4-arm speed sweep --
  setsid bash -c 'RUNROOT=output/fr13_tail6_speedsweep TAG=sw1 SUBSET=subset_b4_sixteen.json \
    SEQUENCE_FILE=scripts/fr13_tail6_speed_sweep_seq.sh bash scripts/fr13_b4_campaign_driver.sh \
    > output/fr13_tail6_speedsweep/driver.sw1.log 2>&1' </dev/null >/dev/null 2>&1 &
(nativemtp5+tail6b+tail6e+tail6c same-session => the definitive tree-vs-native speed verdict + geometry.)

---

## Direction-2 BREAK-EVEN ANALYSIS + clean net-negative result (2026-07-18, loop-driven)

### The clean same-session number (b7, bracketed deploy_speed, subset_b4_sixteen, cache-off):
| arm            | nodes | accept_per_event | s_per_fwd_gpu | derived_tps |
|----------------|-------|------------------|---------------|-------------|
| tail6 (spine)  | 21    | 4.317            | 0.0935        | **5.06**    |
| tail6b (d6/d7) | 25    | 4.500            | 0.1050        | **4.76**    |

Arctic branches raise accept **+0.183** but are **NET-NEGATIVE on tps** (4.76 < 5.06). The +4 nodes cost
+12% verify; +0.18 accept doesn't pay for it. **This kills the "arctic width" lever definitively.**

### Break-even (why): tps ∝ committed/s_fwd. Each tree node adds ~0.0029s to s_fwd (measured: +0.0115s/4
nodes). d(tps)/d(node) > 0 requires **d(accept)/accept > d(s_fwd)/s_fwd**, i.e. accept must rise
**> 0.138 per added node** (= 4.317 × 0.0029/0.0935). Arctic branches deliver **0.046/node** (correlated
siblings, same suffix-match) → guaranteed loss. A 25-node arm needs accept **> 4.97** to beat 21-node
tail6; arctic tops out ~4.5-4.65. NOTE: piggyback (Dir-1) does NOT help — shrinking the step makes the
FIXED per-node verify cost a BIGGER fraction (0.0029/0.20 = 1.45%/node vs 0.0029/0.28 = 1.03%/node now),
so Dir-1 and Dir-2 do NOT simply "multiply" (that claim in the roadmap is wrong for node-adding levers).

### The only source that clears 0.138/node = a DECORRELATED d6 candidate (recovers the 0.334 handoff miss,
~+0.4 accept for 1 node = 0.4/node >> 0.138). But MTP maxes at 5 heads (mtp_near[0..4], mtp_spine pads
with near[-1]); a real d6 model prediction needs **+1 MTP forward** — verify-basis-FREE (drafter, not
s_fwd) but **fullstep-COSTLY** (~+20ms drafter). Fullstep accounting: MTP-d6 ~18.6 tps vs tail6 ~18.75 =
tps-NEUTRAL. No FREE decorrelated tail source exists (arctic is the only free tail proposer, self-correlated).
=> COST-GATE: don't build the risky +1-forward MTP-d6 patcher change for a fullstep-neutral lever.

### CREATIVE lever (the ONE untested tps-positive candidate) = REALLOCATION, not addition.
Zero-node-cost: keep the fixed 21-node budget, move deep-tail nodes (d10,d11 — conditional already
0.90-0.95, low marginal value) to d6 branches (the 0.334 leak). Same 21 nodes => same s_fwd => the accept
delta IS the tps delta. tail6realloc = 15 head + tail spine d6-9 (4) + 2 d6 branches = 21 nodes (==tail6).
NON-MONOTONE (drops d10/d11 => can regress on long-repeat spans) => DIAGNOSTIC, not lossless-shippable;
tells us whether the leak is worth more than deep reach. tail_len is tree-derived (chain follows wide_D)
so a shorter tree re-syncs the drafter WITHOUT a new flag — but must CPU-verify the drafter re-syncs before
running. Gated behind the strategic anchor: only worth building if the tree is even competitive with native.

### ACTION: run the STRATEGIC ANCHOR first (zero-build, zero-drift): native MTP-5 vs tail6 (best tree)
same-session (fr13_native_vs_tail6_anchor_seq.sh). If native wins even vs the best tree, direction-2 is
moot (node-adding can't leap past native). If tail6 competes, build+test the tail6realloc diagnostic.
