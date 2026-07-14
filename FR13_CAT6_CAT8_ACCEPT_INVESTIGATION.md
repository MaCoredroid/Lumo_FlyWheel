# Investigation: why is cat6 accept (3.594) > cat8 accept (3.336)?

**Question (user, 2026-07-13):** cat8 ⊃ cat6, so cat8 should accept ≥ cat6 on matched inputs. Why does the
live run show cat6 > cat8? Real defect or trajectory noise?

## The superset bound (theory)
cat8 = `[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,0,0)]`
cat6 = `[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]`  =>  **cat8 = cat6 ∪ {(0,1),(0,0,1)}** (confirmed).
On IDENTICAL (draft tokens, target logits), cat8 offers every cat6 path plus 2 more => cat8 accepted length
≥ cat6 at every forward. So **cat8 accept ≥ cat6 accept on MATCHED inputs.** The live aggregate violates it.

## Evidence so far (NOT matched, so not decisive)
- Aggregate: cat8 **3.336**, cat6 **3.594** (cat6 − cat8 = +0.258).
- Per-task (same 16 tasks): mean(cat8−cat6) = **−0.167**, cat8 ≥ cat6 on **6/16** (~2.1σ lean to cat6).
- BUT per-task is still NOT matched — temp 0.6 diverges WITHIN a task once tokens differ. Tell: the biggest
  cat6-favoring gaps are on FAILING/meandering tasks (14369 −0.83, 14598 −0.68, 14182 −0.52), while clean
  resolved tasks are ≈0 (12907 +0.05, 13453 +0.07). That pattern = trajectory divergence, not a uniform
  verify defect (a verify defect would be task-independent). Suggestive of noise, NOT proof.

## Two candidate REAL mechanisms (can't rule out without a matched test)
1. **RNG-draw-order (temp 0.6):** cat8's 2 extra branch nodes consume extra drafter sampler draws → shifts
   the spine draft tokens vs cat6 on the same prefix → different accepts (either direction).
2. **Residual M-dependence of the spine verify** (M = node count): spine verify should be M-invariant
   (cat8 M=8 spine == cat6 M=6 spine == native). If imperfect, cat8's spine accepts differently. (The garble
   work made the spine M-invariant for GARBLE / wrong-accepts; the exact accept COUNT is a separate property.)

## The decisive experiment (queued, runs after native frees the GPU)
**Greedy (temp 0) fixed-prompt A/B** — `scripts/fr13_cat6_cat8_accept_bound_exp.sh`:
- Greedy => deterministic drafts (no RNG-order confound) + deterministic target => cat8 and cat6 produce the
  SAME output; the tree only changes HOW MANY tokens commit per forward.
- Boot locked serve with cat8, run a fixed completion prompt at temp 0, read spec_decode accept/forward from
  /metrics delta. Repeat with cat6. Same prompt, same output.
- **Assert cat8 accept/forward ≥ cat6.** If cat8 ≥ cat6 (greedy) => verify-side is M-invariant/clean =>
  the live temp-0.6 cat6>cat8 is RNG-order + trajectory noise (bound holds where it must). If cat8 < cat6
  (greedy, matched output) => REAL structural violation (M-dependent spine verify or accept-logic bug) =>
  localize + fix.
- Caveat: greedy tests the VERIFY-side (M-invariance). If greedy passes but a temp-0.6-specific RNG-order
  effect is suspected, a follow-up single-forward fixed-seed temp-0.6 matched test isolates that.

## Speed metric redesign (user, 2026-07-13): real wall-clock TPS, not derived
`derived_tps_gpu = committed/s_per_fwd_gpu` uses the VERIFY-FORWARD GPU time only => it's an UPPER BOUND
("how fast IF the forward were the only cost"), ignoring drafter+committer+scheduler gaps (~30% per our own
"Tree TPS is overhead-bound" finding). That's why it reads 64-72 while real per-request wall TPS was ~15-17.
Neither existing metric is clean: derived_tps overstates (forward-only); per_request_decode_tps is
prefill-confounded at B>1. Both are contaminated because they're derived from the AGENTIC run.

**Good metric = `decode_tps_wall = N_committed / (t_last_token − t_first_token)` on a CONTROLLED fixed-prompt
benchmark, B=1, temp 0.6 seed 0, driven directly (no agent/offload).** Clean because: wall-clock (nets accept
gain vs tree overhead honestly), B=1 (no co-residency; prefill excluded by measuring from first gen token),
committed-not-drafts (garble-immune). Secondary: aggregate throughput at B=deploy. accept/forward stays a
DIAGNOSTIC (the lever/why), not the headline. Retire derived_tps to "forward-limited ceiling" for diagnosis;
the ceiling−wall gap = our optimization target (committer/replay/drafter overhead).

## Experiment (queued, after native) — ONE controlled benchmark, TWO measurements
`scripts/fr13_cat6_cat8_accept_bound_exp.sh` boots the locked serve per tree (cat8/cat6/native), B=1, fixed
prompt, and records:
1. **Greedy (temp 0) accept/forward** — the superset-bound test. Deterministic drafts+output => cat8 MUST
   accept ≥ cat6. If cat8 < cat6 greedy => real M-dependent verify defect; else live diff = temp-0.6 noise.
2. **temp-0.6 seed-0 `decode_tps_wall`** (committed / decode-wall, from first gen token) — the REAL speed
   number, cat8 vs cat6 vs native, B=1 same boot-era. This replaces derived_tps as the headline.

## Complete same-boot A/B (all 3 arms done, 2026-07-13) — cat8 is SLOWEST, non-monotonic
| arm | resolve | accept | per_forward | derived_tps | committed/per_forward |
|---|---|---|---|---|---|
| native MTP-5 | 8/16 | 3.466 | 0.1995 | 70.94 | 22.4 |
| **cat8 (8-node)** | 8/16 | **3.336** | **0.2173** | 63.92 | **19.9** |
| cat6 (6-node) | 7/16 | 3.594 | 0.1988 | 72.43 | 23.1 |

**BOTH bases agree: cat8 slowest, cat6 fastest, native middle.** cat8 has the LOWEST accept AND highest
per_forward => slower than native's 5-chain. Accept is NON-MONOTONIC in nodes (cat6 6 > native 5 > cat8 8) —
if branches just added accept, cat8 would be highest; it's lowest. Either (a) trajectory noise across 3
different token streams, or (b) cat8's 2 extra branches cost more per-forward than they earn (M-dependence /
drafter shift). This CHALLENGES "branches = speed" and demands the matched experiment below.
NOTE: cat8-vs-native is NOT a clean superset test (different drafter/attn/committer: tree path vs native MTP).
cat8-vs-cat6 IS (same forked tree path) — that's the sharp one.

## 🛑 CONFOUND RETRACTION (2026-07-13): the controlled experiment below ran with FR13_ATTN_KV_REMAP OFF
The accept-bound orchestrator omitted `FR13_ATTN_KV_REMAP=1` (0 ENGAGED in docker_full.log), so **cat8/cat6
probes were GARBLING** => their accept is garble-CONFOUNDED (wrong-accepts). native has no tree => clean
regardless. Impact: **the "cat6>cat8 at greedy => M-dependence not noise" conclusion is NOT cleanly
established** (the greedy run was itself garbling). What SURVIVES: (a) native > trees holds/strengthens (trees
garble-inflated, so clean trees even lower); (b) cat6>cat8 is corroborated by the CLEAN matrix (cat6 3.594 >
cat8 3.336, remap-on 0/84 garble) — but the clean GREEDY M-dependence-vs-noise test is PENDING a re-run.
Root cause fixed in scripts/fr13_cat6_cat8_accept_bound_exp.sh (now sets FR13_ATTN_KV_REMAP=1 +
DEVICE_MULTIDRAFT=1). Re-running clean. TREAT THE NUMBERS BELOW AS GARBLE-CONFOUNDED (cat8/cat6 rows).

## RESULT (controlled experiment, B=1 fixed prompt) — 🛑 GARBLE-CONFOUNDED (remap OFF), see retraction above
| arm | mode | accept/fwd | tps_wall (committed/decode) |
|---|---|---|---|
| cat8 | greedy | **3.368** | 18.9 |
| cat6 | greedy | **3.664** | 20.5 |
| cat8 | temp06 | 3.031 | 16.9 |
| cat6 | temp06 | 3.414 | 18.8 |
| native | greedy | **3.813** | **22.21** |
| native | temp06 | **3.691** | **21.45** |

### FULL RESULT: native > cat6 > cat8 (accept AND wall-TPS, greedy + temp06) — challenges "branches=speed"
On this controlled B=1 fixed-prompt benchmark, NATIVE (MTP-5, no branches) is the FASTEST — highest accept
AND highest wall-TPS. Accept MONOTONICALLY DECREASES with node count: native(5) 3.813 > cat6(6) 3.664 >
cat8(8) 3.368 (greedy). => adding tree nodes REDUCES accept here = strong M-dependence. The branched trees are
NET SLOWER than native on this workload.
TWO effects: (a) within the forked tree path, more nodes = less accept (cat8<cat6, pure M comparison);
(b) even the smallest tree cat6 < native (forked-tree-path overhead / M-dep vs stock MTP).

**HONEST CAVEATS (do not over-generalize from one prompt):**
1. SINGLE PROMPT = predictable code (TTLCache). Predictable content => the SPINE draft usually hits => branches
   (which pay off exactly when the spine is UNCERTAIN) are barely exercised => this prompt UNDERSTATES branch
   value. A diverse/ambiguous-prompt benchmark is needed to fairly measure what branches buy.
2. native is a DIFFERENT code path (stock MTP + FLASH_ATTN) vs forked tree path (TREE_ATTN + tree drafter);
   native-vs-tree mixes path + M. cat8-vs-cat6 is the clean M comparison (and it's monotone worse).
3. BATCH_INVARIANT=0 => M=5/6/8 take different GEMM-autotune paths => numerical M-dependence possible (an
   artifact, fixable), not necessarily fundamental.
4. B=1 (deployment is B4). B=1 is the clean HBM-bound condition where trees SHOULD win most — and they don't.

**CONCLUSION + next lever:** the M-dependence (more nodes -> less accept) is the drag making branched trees
slower than native. This is the directive's step-3 target: an M-INVARIANCE compute-only fix (localize which
L0 sub-op is M-dependent: conv/scan/state/GEMM; the in_proj-pad LUMO_FB_PROJ_PAD covered some but not all)
would restore trees' accept >= native + the branch bonus. WITHOUT it, branches are a net speed LOSS on
predictable content. Garble remains fixed (0%, correctness intact); this is purely accept-rate/speed.
Recommended next: (a) diverse-prompt benchmark to measure true branch value; (b) localize the M-dependent
sub-op (greedy cat6-vs-cat8 accept as the gate) + compute-only fix.

**cat6 > cat8 EVEN AT GREEDY (3.664 > 3.368).** Greedy = deterministic drafts + target, so cat8 ⊇ cat6 =>
cat8 accept >= cat6 MUST hold if the spine is M-invariant. VIOLATED => **NOT trajectory noise** (my leading
hypothesis is REFUTED). It's a **real M-dependence**: cat8's 2 extra branch nodes (M=8) perturb its own spine
(verify or draft) vs cat6 (M=6) => cat8 accepts FEWER correct tokens. Robust: cat6>cat8 across greedy + temp06
+ live run (3.594>3.336). No garble (tokens committed are correct) — this is a SPEED defect (accept rate), not
a correctness one.

**Consequence:** cat8's extra branches are a NET LOSS — M-dependence (lower accept) + bigger per-forward
outweigh the 2 branches' extra acceptance. cat6 (branched 6-node) is the faster tree. Does NOT violate
keep-branches (cat6 keeps its (1,) branch). Likely cause: residual M-dependence the garble in_proj-pad
(LUMO_FB_PROJ_PAD_ROWS) didn't fully cover, and/or GEMM-autotune numerics (BATCH_INVARIANT=0 => M=8 vs M=6
different GEMM path). => directive step 3: an M-invariance compute-only fix would restore cat8 accept >= cat6.
Probe bug caught+fixed mid-run: decode_tps_wall counted SSE chunks not committed tokens; accept was always
correct (/metrics delta); all tps re-derived committed/decode_wall.

## LOCALIZATION (spine-mdep workflow wf_81953c42, 2026-07-13) — FA2 query-tile, fix=FR13_FA2_QPAD
7-reader workflow over L0 sub-ops. Verdicts: in_proj_ba / conv1d / gdn_scan / gating / gemm_batchshape /
out_proj = NOT M-dependent for cat6(M=6)-vs-cat8(M=8); tree_attention = PARTIAL (the carrier).
- 5/7 provably BIT-IDENTICAL M=6-vs-8: padded_nodes=1<<(n-1).bit_length() => M=5,6,8 all N_PAD=8 (scan geom
  identical); in_proj cuBLASLt switches at M>=9 (cat6/cat8 both <=8 => identical a/b). So the carrier is NOT
  "1 of 7" — it's the ONE residual that changes M=6->8.
- ROOT (HIGH): forked-FA2 tree-attn QUERY-TILE. max_seqlen_q=6 vs 8 drives kBlockM=64 partial-tile
  predication (Is_even_MN) + tree-bias q/k offsets. Independent binding FR13_FA2_MAB replay M_DEPENDENT=True;
  depth-ladder bit-exact depths 0-2, first-nonzero depth 3-4 = exactly the accept-truncation margin (accept
  3.4-3.8); MONOTONE in M (native 3.813 -> cat6 3.664 -> cat8 3.368) — a codegen ULP flip would be non-monotone.
- FIX: FR13_FA2_QPAD (pad query tile to fixed M => M-invariant predication; compute-only, no HBM tax). BUILT
  at commit 030a1c22 but NOT in the live patcher (comment only @ fr10_phase4_patch_vllm_tree_gdn.py:14944) =>
  must be RE-LANDED, not flag-flipped. gdn_scan N_ACTUAL codegen = LOW control (N_PAD=8 constant).
- ssm_state reader errored (retry cap); covered by the N_PAD=8 argument (scan/state geometry M-invariant).
PLAN: (1) clean experiment (running, remap ON) causally confirms cat8-spine < cat6-spine (greedy, non-garble);
(2) re-land FR13_FA2_QPAD; (3) gate = greedy cat8-spine accept >= cat6 (M-invariant) AND garble stays 0/undef
AND same-boot A/B. Directive step-3 = DONE localizing (FA2 query-tile, not the guessed conv/scan/state).

## CLEAN RESULT (remap ON, ENGAGED=1 verified, 2026-07-13) — M-dep REAL but SMALL; temp06 branches WIN
| arm | greedy accept | temp06 accept | temp06 wall_tps |
|---|---|---|---|
| native (M=5) | 3.813 | 3.691 | 21.45 |
| cat6 (M=6)   | 3.645 | 3.522 | 19.34 |
| cat8 (M=8)   | 3.558 | 3.673 | 19.34 |

- GREEDY (deterministic): native > cat6 > cat8; cat6−cat8 = **0.087** (confound had said 0.296 = **3.4x
  exaggerated** by garble which DEFLATED cat8). So cat6>cat8 M-dep SURVIVES clean but is SMALL — matches the
  workflow FA2-query-tile depth-3-4 localization. Superset VIOLATED at greedy by 0.087.
- TEMP06 (ship config) FLIPS: cat8 3.673 ≈ native 3.691 > cat6 3.522. Branches ACTIVE at temp>0 => cat8's
  extra branches recover it to native accept, OVERCOMING the small greedy M-dep => "branches=accept" HOLDS at
  ship config. Superset HOLDS at temp06 (cat8>cat6). (1 sample — suggestive.)
- WALL TPS (temp06, the REAL metric): native 21.45 > cat6 = cat8 19.34 (~10%). Even at accept parity the
  tree's bigger per-forward eats it => the real tree-vs-native gap = PER-FORWARD OVERHEAD, not FA2 M-dep.
CONCLUSION: FA2_QPAD would restore the greedy superset (cat8>=cat6) but the benefit is small (0.087) and
temp06 already has branches winning. The bigger lever for tree-vs-native wall-TPS is per-forward overhead.
Re-landing FR13_FA2_QPAD = correct M-invariance (step-3) but low-impact; verify with a QPAD A/B before baking.

## 🛑 QPAD LEVER REFUTED BY GIT HISTORY (2026-07-13) — do NOT re-land
The spine-mdep workflow recommended re-landing FR13_FA2_QPAD (FA2 query-tile). GIT HISTORY OVERTURNS IT:
- 030a1c22 (Jun14): QPAD built.
- **8b7684dd: "FA2-tile-carrier OVERTURNED — QPAD fixed named carrier L31->0.0 but e2e flips STAYED 24 =>
  FA2 query-tile NOT the carrier; first-nonzero is L0 GDN (conv1d prime suspect)."**
- **033c6805: "REJECT A' workflow rec (build FR13_FA2_QPAD) — overturned/stale lever."**
The workflow read the "built" comment but MISSED the two overturn commits => it rediscovered a refuted lever.
QPAD makes the FA2 tile M-invariant but does NOT move e2e. (The overturn was garble-era (24 flips); garble
was ultimately fixed by the attn-KV remap = data movement, so BOTH FA2 and L0-GDN were superseded compute
hypotheses.) => DO NOT re-land QPAD.

## FINAL CONCLUSION (M-dependence investigation)
1. cat6>cat8 accept M-dependence is REAL but SMALL: **0.087 greedy** (confound had 3.4x-exaggerated it to 0.296).
2. At the SHIP config (temp06) the branches WIN: cat8 3.673 ~ native 3.691 > cat6 3.522 => superset HOLDS,
   "branches=accept" holds where it matters.
3. The M-dep carrier is FA2-vs-L0-GDN AMBIGUOUS (workflow said FA2; git overturned FA2->L0-GDN for garble).
   Settling it needs the matched single-forward capture (slice cat8->cat6, first-nonzero layer). LOW priority:
   the effect is small + greedy-only + the fix lever (QPAD) is overturned.
4. The REAL tree-vs-native gap is WALL-TPS ~10% = PER-FORWARD OVERHEAD (bigger verify), NOT the FA2/GDN M-dep.
   That (per-forward cost) is the high-value lever if tree-vs-native speed is the goal.
=> The garble ship-goal is MET (0/undef both trees, resolve~native). The accept-rate M-dep is understood,
small, ship-config-benign, and its named fix is refuted. Recommend NOT chasing QPAD; if pursuing tree speed,
attack per-forward overhead. Directive step-3 "compute-only M-invariance fix": localized but the pre-built
fix is a dead lever; a new fix isn't worth it for 0.087 greedy-only.

## MATCHED PROOF on REAL SWE task (astropy-14598, B=1, 2026-07-13) — M effect on spine CONFIRMED (not batch)
B=1 => batching physically eliminated. cat8(branch-diag)+native, real chat prompt, greedy/temp06/temp10.
| temp | cat8 total | native | cat8-native | cat8 A_spine(~) | A_branch(rescue) |
|---|---|---|---|---|---|
| greedy | 3.589 | 3.526 | +0.06 | ~3.35 | 0.225 |
| temp06 | 3.571 | 3.688 | -0.12 | ~3.34 | 0.225 |
| temp10 | 2.916 | 3.905 | -0.99 | ~2.68 | 0.214 |
FINDINGS:
1. **M effect on spine CONFIRMED, and it is NOT batching** (B=1): cat8-spine (~3.35) < native (3.526) by
   ~0.18 at greedy => intrinsic M-dependence (FA2 query-tile), not co-residency. User's "cat8-spine==native"
   is REFUTED — the spine IS perturbed by M.
2. Branch rescue REAL + stable: ~0.22/fwd (6.4-7.1%) at ALL temps incl temp1.0; all 3 branches fire; deeper
   branch (0,0,1) dominates at temp06. The extra 2 branches (vs cat6) add 0.167/fwd.
3. cat8 ~ native at greedy (rescue 0.22 ~compensates spine deficit 0.18); cat8 << native at temp10 (spine
   deficit blows to ~1.2, rescue can't cover). Gap GROWS with temp.
WHY cat8 NOT > cat6: cat8's +0.167 extra-branch gain ~cancels its extra spine M-perturbation (M=8 vs M=6).
FIX: make the spine M-INVARIANT (FA2 query-tile pad, compute-only no-HBM-tax). Then cat8-spine==cat6-spine
=> cat8 = spine + 3 branches > cat6 = spine + 1 branch (extra 0.167 becomes pure gain). cat6 running now to
confirm cat6-spine > cat8-spine directly (the pure M comparison, B=1).
CAVEATS: instrument undercounts total ~4% vs probe (spine abs ±0.19, used probe-total x instrument-ratio);
native has a different drafter (not a perfect spine match) => cat6 is the clean same-path M comparison.

## PURE-M SPINE CONFIRMED (cat6 vs cat8, same path, B=1, real astropy-14598)
| temp | cat6_spine | cat8_spine | Δ(cat6-cat8) |
|---|---|---|---|
| greedy | 3.560 | 3.175 | +0.385 (warmup-contaminated) |
| temp06 | 3.608 | 3.317 | +0.291 (CLEAN delta) |
=> cat6's M=6 spine accepts MORE than cat8's M=8 spine by ~0.3 (same forked path/drafter, B=1 => intrinsic
M-dep, not batching). CONFIRMS: cat8's larger M perturbs the spine => cat8 not > cat6. cat6_spine (~3.56) ~=
native (3.526); cat8_spine (~3.18) well below => the perturbation jumps steeply M=6->8 (FA2 query-tile 6->8).
CAVEATS (owe a cleaner instrument): (1) branch-rescue instrument UNDERCOUNTS vs probe, variably (cat6 misses
~0.42, cat8 ~0.19) => absolute spine numbers unreliable; the same-temp cat6-vs-cat8 DELTA is valid (same
instrument). (2) greedy = warmup-contaminated (first-mode cumulative snapshot); temp06 delta is clean. (3)
single sample/temp => totals noisy (cat6_tot greedy 4.039 vs temp06 3.470). FIX direction validated:
M-invariant spine => cat8-spine==cat6-spine => cat8=spine+3br > cat6=spine+1br. Fix-front workflow w6gqaot0t
localizing FA2-vs-L0-GDN + designing the capture + compute-only fix.

## FIX-FRONT WALL (2026-07-13): MAB localizer DEAD for fused-conv build; carrier=FA2 (code analysis)
The empirical MAB co-residency localizer (fr13_mab_coresidency_localize.sh) ENGAGE-FAILED 3x (torch.compile
autotune crash w/ device-multidraft; shallow warmup; deep warmup). ROOT CAUSE (code): the FR13_GDN_SUBOP_MAB
hook stashes pre_conv/conv_state on the NON-FUSED conv path, but the ship bakes FR13_TREE_CONV_FUSED=1 (fused
emulation) which BYPASSES those stash points => "conv_state snapshot missing (stash disengaged)" => never
fires. It is a STALE tool for the current fused build (the garble-era conv1d_out 9.77e-4 verdict was a
non-fused build + M10-vs-M5, NOT cat6-vs-cat8).
BEST-AVAILABLE LOCALIZATION (wf#1 code analysis, HIGH): for cat6-vs-cat8 (M=6 vs 8, both N_PAD=8), conv / scan
/ in_proj are M-INVARIANT (padded_nodes=1<<(n-1).bit_length()=8 for M<=8; in_proj cuBLASLt switch at M>=9). So
the directive's "L0 conv/scan/state" premise is REFUTED for THIS regime. The only residual = FA2 query-tile
(max_seqlen_q 6 vs 8), confirmed M_DEPENDENT=True by FR13_FA2_MAB.
FIX = FA2 QPAD, but: (a) NOT in live patcher = 272-line re-port into scripts/fr10_phase4_patch_vllm_tree_gdn.py
(inject into installed flash_attn_interface.py); (b) its ONE empirical test DROPPED accept to 2.643 (below
native) = uncertain it helps the accept-rate. So the fix is a substantial+uncertain investment.
STATE: mechanism confirmed (cat6-spine>cat8-spine ~0.3, B=1); carrier=FA2 (code); localizer dead; fix=QPAD
(big+uncertain). Effect SMALL (0.087 greedy, ~0.3 real). Garble deliverable MET (0/undef, resolve=native).
NEXT OPTIONS: (A) re-port QPAD + A/B (big, uncertain); (B) build a fused-build FA2/GDN localizer; (C) accept
the small M-dep as within-floor (deliverable met) and stop. Surfacing for a priority call given the ROI.

## CORRECTION (2026-07-13): QPAD is NOT refuted for cat6-vs-cat8 (the refutation was CAT9)
Red-team of the QPAD-accept-drop: "QPAD | 9-node | 24 flips | accept 2.643" = the test was on CAT9 (M=9),
whose carrier is L0-GDN (N_PAD=16 vs 8, per NODE7-LADDER first-nonzero L0 linear_attention 2 ULP). QPAD drove
the FA2 carrier L31 + 14/16 layers -> 0.0 but e2e flips STAYED 24 because the L0-GDN seed is UPSTREAM of FA2
(L3) — an FA2 fix cannot remove an L0-born divergence AT M=9. => the refutation is REGIME-SPECIFIC to cat9.
For cat6-vs-cat8 (M<=8, N_PAD=8), GDN is M-invariant (wf#1), so the carrier is FA2 (no L0-GDN seed) and QPAD
is UNTESTED, PLAUSIBLY CORRECT. My earlier "QPAD refuted -> wall" CONFLATED cat9 with cat8. Corrected:
the fix path is OPEN = re-port QPAD (archived 030a1c22 / origin/fr13-fa2-qpad) + A/B on cat8 (gate cat8-spine
>=cat6 + garble0 + lossless-vs-nonspec). Lossless-by-construction (CPU 0.0) so it cannot reintroduce garble.

## COST-GATE STOP (2026-07-13): carrier UNSETTLED, no confident cheap correct fix — M-dep = within-floor
Git lineage shows the FA2-vs-L0-GDN carrier OSCILLATED: 06676346 (query-pad premise CONTRADICTED) -> 9ad6793f
(FA2 IS carrier) -> 8b7684dd (FA2 NOT carrier, L0-GDN upstream); + wf#1 (FA2 for M<=8) vs wf#2 (L0-GDN). It is
GENUINELY UNSETTLED. QPAD fixes only FA2, which 8b7684dd suggests is INSUFFICIENT (L0-GDN seed upstream). The
empirical localizer (MAB) is DEAD for the fused build. => NO plausibly-cheap CONFIDENT correct fix.
Per the cost-gate (speed=goal, STOP if no plausibly-cheap correct path): STOP the fix chase. The accept-rate
M-dependence is SMALL (0.087 greedy, ~0.3 real) and SHIP-BENIGN (temp06 cat8 3.673~native 3.691>cat6; resolve
cat8 8/16=native). GARBLE DELIVERABLE MET (0/undef both trees, resolve=native).
OPEN ITEM (user-prioritizable, not cheap): to actually land a fix, first BUILD a fused-build FA2/GDN localizer
(the MAB is non-fused-only) to definitively settle FA2-vs-L0-GDN, THEN fix the confirmed carrier + A/B. That is
a multi-session effort for a small ship-benign gain — deferred pending explicit priority.
SUMMARY of the whole cat6-vs-cat8 investigation: (1) cat8 not>cat6 = M-perturbation on spine (confirmed B=1,
cat6-spine>cat8-spine ~0.3); (2) branches DO rescue (~0.22/fwd, 6-7%, all temps); (3) carrier unsettled
FA2/L0-GDN; (4) small+ship-benign; (5) garble goal MET.

## QUALITY QUESTION RESOLVED (2026-07-13): M-perturbation is SPEED-ONLY, not quality
User reframe: "is it a quality issue if cat8 didn't accept the right token?" ANSWER: NO — cat8 commits the
RIGHT tokens. EVIDENCE: the token-level garble gate matrix_build 15/15 -> 0/15 + token_ledger 0/15 = cat8
fix-ON commits BYTE-IDENTICAL token sequences to native (a TOKEN-SEQUENCE compare, not just gross-garble).
Native is a correct spec-decode (lossless vs true target by construction), so cat8==native => cat8 LOSSLESS.
The earlier chat-prompt "divergence at char 10" was CONFOUNDED (237-char partial capture, post-turn junk,
native-not-true-target) — flagged, NOT counter-evidence. => the accept-rate M-perturbation LOWERS accept
(speed) but does NOT flip committed tokens (quality intact). Optional extra rigor: cat8 vs NON-SPEC true-target
full-token-id compare (no-spec launcher exists) — DEFERRED (question already answered by token_ledger 0/15).

## FR13 GARBLE SHIP-GOAL: COMPLETE (final state 2026-07-13)
- Garble: 0/undef both branched trees (cat8 0/84, cat6 0/61) on full ship config (branched+cache+SWE-Verified).
- Resolve: cat8 8/16 = native 8/16; cat6 7/16 (-1 flaky 14096). Degradation ELIMINATED.
- Fix: FR13_ATTN_KV_REMAP (attn-KV re-linearization), baked. Structurally complete (all post-commit states remapped).
- cat6-vs-cat8: cat8's larger M perturbs spine (~0.3, B=1); branches rescue ~0.22/fwd; SPEED-only, quality intact.
- Accept-rate M-dep fix: cost-gate-STOPPED (carrier unsettled FA2/L0-GDN, localizer dead, no cheap confident fix).
DEFERRED (user-prioritizable): (a) speed fix needs a new fused-build localizer (multi-session); (b) non-spec
losslessness confirm; (c) dead-code cleanup #19.

## FUSED-BUILD LOCALIZER — LADDER STEP 1: CONV CLEARED (2026-07-13)
Reopened the cost-gated accept-rate M-dep fix (user greenlit). Built FR13_CONV_SUBOP_MAB (default-OFF,
observe-only): re-runs the SAME imported FUSED op (fused_tree_conv_taps_acc + triton silu) on the SPINE-only
sub-window (M_reduced) vs the full-M spine rows, raw int-view threshold 0.0. Corrects the garble-era
GDN_SUBOP_MAB root cause: that MAB re-ran the NATIVE causal_conv1d_update (never populated on the fused ship
build) -> engage-failed 3x. NOT "conv_state is None". Confound-dodged: passes PRE-splice _fr10_acc (the :3356
native-spine index_copy_ overwrites _fr10_out, not _acc), recomputes silu fresh.
- Engage-fail #1 (self-caught): the A/B .item()-syncs under CUDA-graph capture at EngineCore init -> poisoned
  capture -> boot rc=2. FIX (per established eager-only-diagnostic pattern): capture-guard at helper top +
  boot ENFORCE_EAGER=1. conv-taps M-invariance is regime-independent (same kernel eager/captured).
- CLEAN RUN (run_eager, rc=0): 768 events / 48 distinct layers (16/layer), cat8 served tree_n=9, spine m_red=6,
  deep=8. SUM taps_mismatch=0, SUM out_mismatch=0, ANY-mismatch events=0. Engaged (NOT vacuous), varied across
  a real 256-tok temp06 decode (accept/fwd=3.06, 63 drafts).
=> VERDICT: fused conv (taps + silu) is M-INVARIANT on GB10 -> DEFINITIVELY CLEARED as the accept-rate carrier.
   Matches a-priori (elementwise mul + per-col fp32 add, no cross-row reduction) but now PROVEN on-device.
NEXT (ladder): FA2 query-tile MAB (generalize FR13_FA2_MAB cat9->live-tree, cat8 M=9-vs-M=6), then scan
N_ACTUAL constexpr (6-vs-8 at fixed N_PAD=8). in_proj analytically excluded (cuBLASLt switch at M>=9; M=6,8
share the GEMM path). commits: f03baa2a (localizer) + bbd0ea2f (capture-guard+eager).

## FUSED-BUILD LOCALIZER — LADDER STEP 2: FA2 QUERY-TILE = THE CARRIER (2026-07-13)
Generalized FR13_FA2_MAB (was hardcoded cat9) to cat8. FIRST run was CONFOUNDED (silent
cat9 fallback: TreeAttentionMetadata carries tree_attn_bias but NOT fr10_tree_path0_nodes,
so spine_nodes=None -> cat9 [0,1,2,4,6]/deep6 -> garbage raw_max_abs=2-8 from wrong ancestor
set). Caught by the tell (real query-tile M-dep is ~1-ULP not 8.0) + spine!=live-tree. FIX:
derive spine from tree_attn_bias itself = deepest-node ancestor set (data-driven mask split
0.5*bias.min(), robust to -inf/finfo.min/-1e9, CPU-verified 5/5) + reader confound guard.
- CLEAN RUN (run_eager2, rc=0): 256 events/16 full-attn layers. spine=[0,1,3,5,7,8] deep=8
  (CORRECT, guard passed), bias=[-inf,0.0], recall_vs_served=0.0 (M=9 arm == live output,
  self-check passes). deep-spine raw_max_abs: 16/16 layers NONZERO, worst 6.25e-2 (=1 bf16 ULP
  @ mag~8), typical ~1e-3. NOT the confounded 2-8.
=> VERDICT: forked FA2 query-tile IS M-DEPENDENT (~1-ULP kBlockM query-occupancy, q_offset=
   max_seqlen_q-rows per FR13_FA2_MDEPENDENT_BIND) on cat8's deep spine, all 16 full-attn layers.
   conv=0 + FA2=nonzero => FA2 is THE (likely dominant) accept-rate carrier. scan N_ACTUAL still
   untested but a-priori M-invariant (N_PAD=8 both; only constexpr dead-code residual).
NEXT: validate FR13_FA2_QPAD (pad query to fixed M so max_seqlen_q/kBlockM occupancy is
M-invariant) IN the observe-only MAB (pad both arms -> raw_max_abs->0?) BEFORE touching the live
ship FA2 path. Note FR13_FA2_QPAD is only a COMMENT today (:15097) — not implemented. QPAD was
"refuted" only for the cat9 22-flip GARBLE (L0-GDN carrier), UNTESTED for this cat8 FA2 accept
carrier. commits: cdef5bdc (confound fix). GATE for the eventual live fix: greedy cat8-spine ≥
cat6-spine AND lossless vs NON-SPEC AND garble 0/undef temp06.

## FA2 QPAD (query-pad) REFUTED — carrier is KV/SUFFIX-WIDTH, not query tile (2026-07-13)
Validated the FR13_FA2_QPAD (pad query to fixed max_seqlen_q) hypothesis IN the observe-only MAB before
touching the live kernel. Two iterations of red-team:
- back-pad (dummy rows at BOTTOM): M9-vs-M6 GREW to 1.46e1 -> looked like refutation, BUT self-check
  qpad_self(padM9-vs-unpadM9)=65.0 => back-pad CORRUPTS real rows (forked FA2 is BOTTOM-aligned,
  q_offset=max_seqlen_q-rows). The "refutation" was a BROKEN-TEST ARTIFACT, not real. Caught by self-check.
- front-pad (dummy at TOP, real at BOTTOM, out[-m:]): qpad_self=0.000 at pad 16/32/64 (real rows PRESERVED,
  valid test) AND deep node lands at pad_to-1 in BOTH arms (same tile pos). YET M9-vs-M6 = 6.25e-2 at
  EVERY pad_to == the UNPADDED value. Query padding has ZERO effect.
=> QPAD GENUINELY REFUTED for cat8 (valid test): pinning max_seqlen_q does NOT make FA2 M-invariant.
   The carrier is NOT the query tile -- it is the KV/SUFFIX WIDTH (M9 has 9 suffix keys + 9-wide bias,
   M6 has 6). Even though the deep spine node bias-masks the branch keys (-inf), their PRESENCE changes
   the flash online-softmax block iteration => ~1-ULP (6.25e-2 = 1 bf16 ULP @ mag~8). Exact-same value at
   all query-pads confirms query-independence.
NEXT: test KV-SUFFIX-PAD in the MAB (pad suffix KV+bias to fixed width with -inf-masked dummy keys; both
arms same width -> same block iteration -> M-invariant?). Near-no-HBM-tax (suffix << cached context). If
validated -> live fix (metadata-level: pad tree_attn_bias + suffix KV, no compiled-kernel change). If
refuted -> RESEARCH workflow for other compute-only fixes before any cost-gate. commits: 61e00a26 (front-pad).

## KV-SUFFIX-PAD test BROKEN (self-check caught it) — need kernel-internal understanding (2026-07-13)
Tested KV-suffix-pad (pad suffix KV+bias to fixed width 16/32 with -inf dummy keys). Self-check
kvpad_self(kvpadM9-vs-unpadM9)=0.82 typical / 16.3 worst >> 1-ULP => the -inf dummy keys are NOT
math-neutral in the forked kernel; they corrupt the real M9 row. So kvpad M9-vs-M6 (0.82/18.75) is an
ARTIFACT, NOT a KV-pad refutation/validation. softcap=0.0 (ruled out the softcap-clamps-inf hypothesis).
Likely mechanism: the forked FA2 aligns tree_bias to a WINDOW tied to max_seqlen_q (not the full suffix),
so extending the KV past max_seqlen_q misaligns which keys are masked -> real keys wrongly attended.
STATE of the FA2 carrier: conv CLEARED; FA2 query-tile CONFIRMED carrier (6.25e-2 = 1 bf16 ULP, 16/16
layers); QPAD genuinely refuted (query dim irrelevant, valid front-pad self=0); carrier = SUFFIX-WIDTH
(masked branch keys change flash block iteration); KV-pad test needs correct dummy construction which
requires understanding the kernel's tree_bias/mask/causal alignment. Both re-call padding tests (query
back-pad, KV -inf-pad) broke on kernel-internal conventions -> LAUNCH read-only research workflow to read
the forked FA2 source (fr13_patch_fa2_tree_bias.py + varlen_fwd_tree_bias .cu) and adversarially verify
compute-only no-HBM-tax fixes BEFORE any cost-gate. commits: 20ad27bd (kv-pad arm).

## RESEARCH (wf_6e41fd42): FA2 M-dep MECHANISM CLOSED + fix A' (contiguous-spine reorder) (2026-07-13)
Read-only workflow read the forked FA2 source (scripts/fr13_patch_fa2_tree_bias.py + vendored FA2
mask.h/block_info.h), VERIFIED against source (I spot-checked lines 42/57-58/200/213/29).
ROOT CAUSE of both broken padding tests: the kernel anchors BOTH the tree_bias column window AND the
causal diagonal to `context_len = actual_seqlen_k - actual_seqlen_q` (block_info-derived, NOT tree_bias.shape).
KV-suffix-pad raises actual_seqlen_k without matching actual_seqlen_q => anchor slides => real keys lose their
-inf mask (self=16.3). QPAD front-pad self=0 because k_offset (line 213) compensates when max_seqlen_q>cols.
=> ALL re-call padding fixes exhausted (KV-pad desyncs anchor; to hold anchor you must pad query too = QPAD, refuted).
MECHANISM of the 6.25e-2 (= exactly 1 bf16 ULP @ mag~8): FP REASSOCIATION of the online-softmax butterfly
reduction. Interleaved branches place surviving spine keys in DIFFERENT score-tile columns (cat8 spine cols
{0,1,2,4,6} vs spine-only {0,1,2,3,4}) => different lane partials (col=nj*8+(lane%4)*2+j) => different
Allreduce<4> association tree => 1 ULP. NOT a masking bug (exp2(-inf)=0 exact; recall_vs_served=0).
=> ONLY lever: make surviving spine keys occupy IDENTICAL columns regardless of branch count.
FIX A' (RANK 1, mechanism-guaranteed, adversarial-verified): call-site-local CONTIGUOUS-SPINE REORDER.
Permute suffix q/K/V + both tree_bias axes to pi=[spine-first (depth order), then branches (topological)],
call the fork, UN-permute output. No .cu recompile (arg reorder of 6-16 suffix rows); no-HBM-tax (context
untouched); lossless (exact relabeling; masked branch cols neutral); carrier-B-safe (local to FA2 call, GDN/
committer see original order). VALIDATABLE in the MAB re-call harness BEFORE live: reorder arm -> deep_pi vs
row_m5 == 0.000 (was 6.25e-2) + relabel-neutrality self-check (non-deep within floor) + identity(interleaved)
baseline stays 6.25e-2 as negative control. Refuted/dead: KV-pad, bias-only reorder, separate spine call
(HBM tax), precision-widening, QPAD, BV geom-match, chain5. Scan N_ACTUAL deferred (FA2 = only carrier).
NEXT: implement FR13_FA2_MAB_REORDER arm; if deep_pi-row_m5==0 => promote A' to live call-site-local reorder
in fr13_patch_fa2_tree_bias.py tree-decode wiring (flag-gated) + gate same-seed byte-identity + accept + lossless
+ garble 0. commits: (this doc) research finding.

## FIX A' FULLY VALIDATED (whole spine bit-exact) — 2026-07-13
MAB run_reorder2: contiguous-spine reorder pi=[spine depth-order]+[branches topological]=[0,1,3,5,7,8,2,4,6]:
- deep_vs_M6 = 0.000e+00, WHOLE-spine_vs_M6 = 0.000e+00 (EVERY spine depth bit-exact vs spine-only, 256 ev/16 layers)
- deep_vs_M9(fix magnitude) = 6.25e-2 (cat8 deep moves 1 ULP to the spine-only truth)
- nondeep_relabel = 0.125 = ~1 ULP @ mag16 (branch nodes legitimately shift ~1ULP; NOT corruption — anchor-slide gave 16.3)
- negative control (interleaved identity) stays 6.25e-2 (contiguity IS the property)
=> A' is the mechanism-guaranteed, lossless (exact relabel), no-HBM-tax fix. NUMERICS PROVEN.
LIVE-IMPLEMENTATION COMPLICATION (red-teamed BEFORE coding): the live FA2 decode call
(fr13_patch_fa2_tree_bias.py:570) uses PAGED KV (key_cache/value_cache + block_table), NOT the dense
contiguous KV the MAB re-call used. A' needs spine KEYS at contiguous score-tile columns => must reorder the
suffix KV, which for paged KV touches cache slots the COMMITTER + GDN read later = carrier-B surface. Options:
(1) global spine-first tree DEFINITION (KV naturally contiguous, needs carrier-B audit of positional consumers:
GDN conv leaf-map, committer col-0); (2) call-site-local paged-KV suffix reorder (slot_mapping or dense-suffix
hybrid). Next: gauge carrier-B surface + design live impl; gate greedy cat8-spine≥cat6-spine + lossless vs
NON-SPEC + garble 0/undef temp06.

## LIVE-IMPL DESIGN (wf_5370146c): DENSE-SUFFIX HYBRID, carrier-B-free (2026-07-13)
Design workflow (verified vs source) DECISION: implement FIX A as a CALL-SITE-LOCAL dense-suffix hybrid,
NOT a global reorder. Global spine-first is blocked by 3 vLLM-CORE positional hardcodes we don't own (eagle
drafter depth counters; _get_depth_counts + _prepare_tree_attn_bias BFS builders that build the very bias we'd
permute) + root@flat-0 invariant => largest surface, re-touches locked pipeline. slot_mapping permute = carrier-B
garble. per-call block_table permute = infeasible (>=16-tok block granularity; 9 tree nodes in one block).
DENSE-SUFFIX HYBRID (flag FR13_FA2_SPINE_REORDER default OFF, in fr13_patch_fa2_tree_bias.py :570 decode block):
(1) CONTEXT call: paged, causal=False, tree_bias=None, seqused_k=seq_lens-tree_n (drop suffix), return_softmax_lse;
(2) SUFFIX call: DENSE over key/value[:num_decode] (fresh current-step K/V in scope, no cache gather), permuted
spine-first (q/k/v + both tree_bias axes), causal=True + tb_p, cu_seqlens_k=cu_tree, return_softmax_lse;
(3) un-permute suffix out+LSE, merge_attn_states(output, ctx_out, ctx_lse, suf_out_u, suf_lse_u).
Specialization of vLLM's shipped cascade/DCP split (flash_attn.py:854-915) + merge_attn_states (shipped Triton).
CARRIER-B CONSUMERS TO TOUCH: NONE (permutation confined to local temporaries; cache/slot_mapping/committer/GDN
read natural flat order; output un-permuted before return). BRANCH nodes: attend context(paged,unchanged) + spine
ancestors(contiguous) + self => shift ~1 ULP (nondeep_relabel=0.125), NOT corrupted; MUST gate branch-rescue accept
+ branch commits (user red-team). UNCERTAINTIES to verify: (i) flash_attn_varlen_func returns LSE on the tree_bias
branch when return_softmax_lse=True (fr13_patch_fa2_tree_bias.py:411-421); (ii) native reshape_and_cache write order
(container-only). GATE: hybrid(no-permute)==current single call within-floor; hybrid(permute) cat8-spine==cat6-spine;
lossless vs NON-SPEC; garble 0; accept>=cat6 INCL branch-rescue; speed (2 calls + merge overhead measured).

## GATE-1a FAIL: cascade split is BUGGY (2026-07-13)
Wired FIX A' live (dense-suffix hybrid, FR13_FA2_SPINE_REORDER, module fr13_fa2_spine_reorder.py CPU-tested).
Gate-1a (fr13_fa2_reorder_gate.sh, MODE=0 baseline vs MODE=2 split-only, ENFORCE_EAGER=1):
- MODE=2 engagement marker=1 (hybrid ENGAGED, NOT a silent fallback — wiring/anchor OK).
- BUT split-only is NOT lossless: baseline accept 3.063 -> split 3.625; MODE=2 output = DEGENERATE garbage
  ('\nuser\n<think>\n\n</think>' looping). Inflated accept = garbage is trivially accepted. output_text
  byte-compare was useless (SSE-truncation: m0 captured 2 chars, m2 52 chars) — accept+degeneracy is the signal.
=> The cascade split (context paged causal=False no-bias suffix-excluded + suffix dense causal+bias + merge_attn_states)
   is STRUCTURALLY BROKEN (not ~1ULP), likely the CONTEXT attention (degenerate = model lost context) or the
   merge LSE. Structurally it matches the shipped cascade (flash_attn.py DCP path :900-980 + cascade_attention
   :1127); bug is a subtle detail. NEXT: self-check mode FR13_FA2_SPINE_REORDER=3 (single ref to output + split
   to temp + log max_abs + LSE/component norms) to bisect. Permute already MAB-proven; ONLY the live cascade split
   is broken. Gate working as intended (caught before ship).

## LIVE FIX COST-GATE: dense-suffix hybrid double-rounds; fp32-out unsupported (2026-07-13)
Multi-forward self-check: cascade split is ~1 bf16 ULP RELATIVE on ALL 30 forwards (max_abs/single_absmax
~4-6e-3) = DOUBLE-ROUNDING (single accumulates fp32->bf16 ONCE; split rounds ctx_out+suf_out to bf16
SEPARATELY then merges again). fp32-intermediate fix REFUTED: flash_attn_varlen_func requires out.dtype==q.dtype
(RuntimeError 'Output must have the same dtype as inputs') -> can't get fp32 partials -> double-rounding is
FUNDAMENTAL to the two-pass merge. And the ~1 ULP is FATAL for spec-decode: MODE=2 (split driving) garbled/
degenerate at GREEDY (which normally HIDES garble) = wrong-accept avalanche, not within-floor.
LIVE-IMPL PATHS all blocked/expensive: (1) dense-suffix hybrid = double-rounding (this); (2) slot_mapping /
global spine-first tree = BIT-EXACT but needs vLLM-CORE surgery (eagle drafter proposes BFS/depth-monotonic;
spine-first is NOT depth-monotonic -> patch propose_tree + _get_depth_counts + _prepare_tree_attn_bias) +
carrier-B audit -> high effort/risk for a SPEED-ONLY ~10% gain.
=> EARNED COST-GATE (built the cheapest path, hit a fundamental wall): the 6.25e-2 spine M-dependence is a
WITHIN-FLOOR SPEED RESIDUAL. Garble ship-goal already DELIVERED (FR13_ATTN_KV_REMAP). FIX A' is fully
validated NUMERICALLY (MAB whole-spine bit-exact) + mechanism fully understood (butterfly reassociation) +
live infra committed flag-gated FR13_FA2_SPINE_REORDER (default OFF, ship byte-identical) for future if the
vLLM-core path is invested in or fp32 flash-out lands. NOT a premature no-go: research done, fix built+tested.

## NEW-ANGLES RESEARCH (wf_39c0c4cc, web+code): A1 two-K-source fused kernel (2026-07-13)
User overturned the cost-gate; web+code research (Horace He/TML "Defeating Nondeterminism" 2025, SGLang/FlashInfer
determinism, DASH ICLR26, + vendored FA2 source). MECHANISM confirmed exactly: reduction order is keyed by PHYSICAL
COLUMN (lane (lane%4)*2 butterfly + MMA k-slice); masked branches = exact 0.0 + row_max branch-invariant => CONTEXT
contribution ALREADY bit-identical; carrier = STRICTLY the 2 suffix reductions (row_sum butterfly + acc_o P·V).
DEAD (searched+refuted): split+merge_attn_states (double-round + context re-assoc, fp32 doesn't rescue); bias-column
edit (category error — bias adds to score, can't move columns); column-compaction (P·V shared GEMM, per-row masks);
num_splits=1/BATCH_INVARIANT (wrong carrier = split-count not intra-tile column). NO no-recompile fix survives.
RECOMMENDED = A1: modify the vendored FA2 fused kernel to read TWO K sources in ONE online-softmax pass — paged
context (natural) THEN dense spine-first suffix — single fp32 accumulator, one bf16 round (NO double-round, NO
merge). Context bit-identical (columns unchanged); suffix canonical (spine-first) => spine M-invariant. Register-only,
no-HBM-tax, no carrier-B. Cost: ONE FA2 rebuild. GO/NO-GO before recompile = CPU fp32 online-softmax model proving
split-block context bit-identity + A1-spine==spine-only. NEXT: write that CPU model (no GPU, no recompile).

## A1 REFUTED (CPU go/no-go) + delivery path clarified (2026-07-13)

**A1 (two-K-source fused) REFUTED before any recompile.** `scripts/fr13_fa2_a1_cpu_model.py`
split-block gate: reading the suffix as a SEPARATE dense block (A1's premise) perturbs the
CONTEXT by ~1 bf16 ULP for EVERY row (spine AND branch) purely from online-softmax
rescale timing — `[block-split ALONE, natural suffix] vs single_real = 9.766e-04`, the SAME
magnitude/character as the hybrid double-round. This is a faithfully-modeled fp32 property
(rescale timing, not the butterfly), so it is airtight. A1 does NOT give a proper fix; it
relocates the ~1 ULP from the merge to the block boundary.

**Delivery path = CACHE SLOT REORDER (no recompile, no block-split).** The live tree-verify
reads the suffix KV PAGED (`block_table` + `slot_mapping` + `tree_attn_bias`, tree_attn.py
build()). Permuting the tree-suffix `slot_mapping` spine-first writes the KV spine-first in
the cache; the paged read then processes context-tail + spine-first-suffix in the NATURAL
block structure (one rescale, suffix canonical) — exactly the MAB-proven clean reorder, with
NONE of A1's block-split. Requires (1) permute tree-suffix slot_mapping spine-first, (2)
permute tree_attn_bias columns to match, (3) permute committer/GDN read indices consistently
(carrier-B audit). NO FA2 recompile.

**GATE BEFORE delivery (user: "both spine AND branch proper fix"):** MAB `branch_check`
(`branch_coresident_max`) — each branch's output in full cat8 vs (spine + that branch only).
Spine M-invariance already MAB-proven (`spine_all_vs_m6_max==0`); the branch gate settles
whether the reorder also M-invariantizes branches or only ~1 ULP within-floor. RUNNING:
output/fr13_fa2_mab/branchfix_*. Parser: scripts/fr13_fa2_mab_branch_verdict.py.

## BRANCH GATE + research red-team (2026-07-13, loop)

**BRANCH GATE (GPU, branchfix_20260713T211515Z, 96 events/16 layers):** the A' reorder fixes
the SPINE bit-exact (`spine_all_vs_m6_max=0.0`) but does NOT fix BRANCHES:
`branch_coresident_max=1.250e-01` — node 2 (depth1) 0.0 bit-exact; node 4 (depth2) 1.95e-3
within-floor; node 6 (depth4, anc=[0,1,3,6]) **0.125 GROSS**. Mechanism: a branch's ancestors
are a SUBSET of the spine, so non-ancestor spine nodes stay masked-between its columns AND its
OWN self-column shifts with co-resident branches (col6 solo -> col8 with 2,4 present). Effect
SCALES with branch depth. So the reorder just MOVES branches to different gapped columns.
=> reorder alone under-fixes branches. The M-dep is CONCENTRATED in the deepest/rarest branch
(shallow branches fine) => small net speed impact, but per user requirement branches need a
FIXED-SLOT canonical layout, not just spine-first.

**RESEARCH RED-TEAM (wf_39c0c4cc-3c3):** recommended Angle A1 (single-pass paged-context +
dense spine-first suffix) as "context-bit-identical". REFUTED that precision: the CPU model
with a REALISTIC block (C=500 unaligned, tail 116 + suffix 9 = 125 in one deployed block)
shows A1's separate-suffix-block perturbs context by 9.77e-4 (~1 bf16 ULP, rescale timing).
A1 is context-WITHIN-FLOOR-maybe, NOT bit-identical. Research correctly refuted: Angle C
(split+merge, double-round+re-assoc), Angle D (num_splits=1, wrong carrier), bias-column edit
(category error — bias adds to score, can't move columns), uniform column-compaction (P·V
shared GEMM, per-row masks — impossible).

**SYNTHESIS — the fix that handles BOTH spine AND branch, context-bit-identical:**
Only the CACHE SLOT REORDER (write tree suffix to FIXED CANONICAL slots: spine-first cols
0..S-1 + branches at FIXED cols by node id) is simultaneously (a) single fused call => context
BIT-IDENTICAL (no block-split), (b) spine contiguous => bit-exact, (c) branches at fixed slots
=> M-invariant, (d) NO recompile. Cost = carrier-B consumer audit (committer/GDN reads follow
slots). The carrier-B-free recompile options (A1/A3) all carry the block-split ~1 ULP AND need
a substantial CUDA rebuild (A3 also needs explicit context_len — padding suffix width slides
the `context_len = seqlen_k - seqlen_q` anchor, the KV-pad refutation). NEXT: read-only audit
of carrier-B consumers to scope the cache slot reorder.

## RED-TEAM REFRAME: branch 0.125 is a WITHIN-FLOOR lossless residual, not a production leak (2026-07-13)

The branch gate's `branch_coresident_max=0.125` (node 6) compares full cat8 vs a (spine + node6
only) tree. But PRODUCTION is ALWAYS cat8 (fixed parent array, tree_n=9) — the reduced "solo"
config NEVER occurs. So node 6 is STABLE in production; 0.125 is its distance from a config that
doesn't happen. Node 6 sits ~2 bf16 ULP off its canonical value because its ancestors {0,1,3}
are gapped by masked spine nodes — a within-floor LOSSLESS residual, NOT a cross-tree speed leak.
The systematic "cat8-spine accepts 0.3/fwd less than cat6" leak is SPINE-driven (the 6.25e-2 spine
carrier); the reorder fixes THAT bit-exact. The reorder LATERALLY moves node 6 (~2 ULP, unbiased),
doesn't degrade it.

**Consequence:** branches likely need NO separate fixed-slot layout. The correct "both spine AND
branch" gate = cat8-WITH-REORDER lossless vs NON-SPEC @ temp06 (validates spine fixed + branches
not degraded, in one live gate). Separate branch fix only if that gate fails on branch tokens.
This makes delivery MUCH more tractable: just deliver the SPINE reorder (cache-slot-reorder if the
carrier-B audit says tractable, else A1 whose block-split ~1 ULP is now plausibly within-floor by
the same argument) + losslessness gate. Pending: carrier-B consumer audit (agent, read-only).

## CARRIER-B AUDIT: slot-level reorder is TRACTABLE, N=3 edits (2026-07-13, VERIFIED)

Read-only audit + my spot-check vs source CONFIRM: a slot-level canonical reorder (permute only
the tree-suffix rows of the ATTENTION-group slot_mapping spine-first; keep the BFS tree DEFINITION)
threads pi consistently and is NOT blocked by a BFS hardcode. Two disjoint address spaces:
  - Attention KV addressed by slot_mapping.
  - GDN conv/ssm state addressed by spec_state_indices = mamba_get_block_table_tensor (gdn_attn.py
    :482,:567) — a SEPARATE bank; permuting attention slots never touches it. VERIFIED.
The rejection-sampler/committer is node/token-indexed ("No float, no reduction, no reorder",
patcher :17471-74), invariant to slot permutation. VERIFIED. Eagle drafter is INVISIBLE (slot-level
keeps BFS node numbering — the crux vs the refuted global-tree-reorder that RENUMBERED nodes).

**N=3 edit sites:** (1) derive+apply pi to tree-suffix rows of slot_mapping at construction
(fr10_phase4_patch_vllm_tree_gdn.py ~:10733/:10747-10789, tree structure in hand); (2) column-
permute tree_attn_bias by pi (key axis only; helper spine_first_perm exists fr13_fa2_spine_reorder
.py:54,86); (3) THE ONE REAL EDIT — launch_attn_kv_linear_remap (fr10_gdn_tree_kernel.py:461-466):
src_slot=slot_mapping[qsl+accepted] auto-threads on the permuted map, but dst_slot must use the
UN-permuted/contiguous map so the next forward reads its committed prefix linearly. gather-then-
clone at :478 already makes src/dst overlap safe. Honest caveat: site 3 is a real correctness
dependency (not mechanical) — forget it and the committed prefix scatters to canonical slots.

=> The over-conservative "carrier-B" claim in fr13_fa2_spine_reorder.py:10-21 (which drove the
dense-suffix-split that double-rounds) is REFUTED. The cache-slot-reorder is the clean deliverable:
context bit-identical (single fused call, no block-split), spine bit-exact, branches at fixed
canonical slots, NO recompile. Next: implement N=3 behind a flag (default OFF), gate on LIVE SWE
cat8-vs-E5 (spine argmax==E5 + bag-TV<=0.0593 + accept>=3.076 + branch-lossless) — the goal gate.

## SLOT-REORDER FINAL DESIGN v2 (2026-07-13) — N=5 sites, all Python, NO recompile

Red-team of the audit's "N=3" against live source found TWO missed hazards; design v2 clears both:

**HAZARD 1 — in-kernel causal mask (audit missed; would break BRANCHES):** served decode call is
`causal=True` (fr13_patch_fa2_tree_bias.py:587) and the kernel applies mask AFTER apply_tree_bias
(:139) → key cols permuted while query rows stay BFS ⇒ branch self-KV (e.g. node2→col6>row2)
causally masked. RESOLUTION: causal is PROVABLY REDUNDANT in the decode tree call — (context cols:
all precede all tree rows, never masked; suffix cols: BFS ancestry ⊆ {≤row}, bias already −INF
beyond) ⇒ `causal=False` is byte-identical today AND unlocks any suffix layout. Non-causal template
instantiation already compiled in the fork; apply_tree_bias inserted at source covers both. NO
recompile. (Prefill calls :504/:541/:568 stay causal=True.)

**HAZARD 2 — drafter slot_mapping leak (audit partially missed):** eagle.py:469-470 COPIES
`common_attn_metadata.slot_mapping` into `self._slot_mapping_buffer` for the DRAFT model's KV
writes (drafter gets `spec_decode_common_attn_metadata = cm`, gpu_model_runner:2296-2301) ⇒ naive
in-place permute scrambles the draft cache. Also: fresh permuted CLONES break FULL-cudagraph
pointer stability (graph captures the persistent buffer pointer; deploy=graph). RESOLUTION:
**permute the persistent buffer IN PLACE (pointer-stable, graph reads contents at replay) +
RESTORE after the last verify-side consumer and BEFORE drafter.propose**. Order per step:
build-metadata (permute) → verify forward/graph-replay (reshape_and_cache + FA2 read use permuted)
→ sample_tokens (ATTN_KV_REMAP src auto-threads on permuted; dst via pi) → RESTORE → propose
(drafter sees flat). Restore is stream-ordered (torch op queues after remap kernel) ⇒ safe.

**THE 5 EDIT SITES (flag FR13_SLOT_REORDER, default OFF):**
1. PERMUTE — gpu_model_runner `_build_attention_metadata`, OUTER per-kv-group loop (once per kv
   group per step; NOT inside `_build_attn_group_metadata` which runs per attn_gid → double-permute
   hazard). Gate: full-attn spec type only (mamba/GDN untouched), `use_spec_decode`, per-req
   `self.num_decode_draft_tokens.cpu[r]==tree_n-1` AND span==tree_n (exact spec signal, in scope),
   skip `for_cudagraph_capture`. In-place: `sm[qsl_r:qsl_r+n] = sm[qsl_r:qsl_r+n][pi_inv]`
   (sm_new[j]=orig[pi_inv[j]] ⇒ col k holds node pi[k]). Stash self._fr13_sr = (buffer, spans, pi).
   Metadata-cache path safe: update_block_table passes the same permuted buffer.
2. BIAS — TreeAttentionMetadataBuilder init: `self.tree_attn_bias = bias[:, pi]` ONCE (key axis;
   rows stay BFS). NEVER per-step (metadata caching double-permutes). pi derived from the SAME
   sorted-BFS algorithm text as the runner (both parse SPEC_CONFIG env; assert pi[0]==0; both log
   pi at boot; gate script asserts equal — fail-loud vs divergence).
3. COMMITTER DST — launch_attn_kv_linear_remap (fr10_gdn_tree_kernel.py:461-466): src auto-threads
   (reads permuted map); dst must be FLAT: dst_slot = sm_permuted[qsl + pi[dst_off]] (identity
   sm_new[pi[k]]==orig[k] ⇒ NO extra stash needed, just pi). Thread pi from self._fr13_sr.
4. RESTORE — after ATTN_KV_REMAP apply in sample_tokens, before drafter.propose: un-permute the
   stashed spans (sm[span] = sm[span][pi] — inverse of edit 1). Drafter + all later consumers see
   flat. (eagle's copy_ at :469-470 happens at propose ⇒ sees restored ✓.)
5. CAUSAL — fr13_patch_fa2_tree_bias.py:587 decode replacement: causal=(FR13_SLOT_REORDER!="1").
   Stage-gate: FR13_TREE_CAUSAL_OFF=1 alone first — same-seed probe MUST be byte-identical to ship
   (proves the redundancy claim on real HW before the reorder rides on it).

**Cleared invariants:** RoPE positions row-attached (travel with K into permuted slot) ✓; rejection
sampler node-indexed ✓; GDN spec_state_indices = separate mamba bank ✓; APC unaffected (committed
prefix stays flat via edit 3) ✓; compute_slot_mapping fully rewrites spans each step (no cross-step
accumulation) ✓; eager + FULL-graph both safe ✓. CPU logic model
(scripts/fr13_slot_reorder_logic_model.py): GO — spine cols [0..d] contiguous IDENTICAL cat8/cat6
⇒ M-invariant by construction; committer src-perm/dst-flat verified, scatter hazard demonstrated.

**Staged validation:** S0 patcher self-test (pristine-copy diff) → S1 causal-off byte-identity →
S2 reorder ON: engagement markers + MAB spine bit-exact + garble gate 0 + lossless-vs-nonspec
temp06 → S3 GOAL GATE: matched-proof cat8-spine vs E5 (expect ≈3.53 == native) + branch-rescue
intact + cat8 TOTAL > native (the deliverable).

## S1 IN-PROCESS GATES: PASS (2026-07-13/14, s1_causal_kperm run, 256 events / 16 layers)

- **CAUSAL arm = 0.000e+00 int-exact** (causal=False vs causal=True on identical live operands,
  every event, every layer) — the causal-redundancy claim is PROVEN ON HW. Edit 5 is safe.
- **KPERM arm (exact delivered-fix semantics: BFS queries + KEY-only spine-first perm +
  bias[:,pi] + causal=False): kperm_spine_all_vs_m6_max = 0.000e+00** — the WHOLE spine is
  bit-exact vs spine-only under the fix AS DELIVERED. The op-level M-invariance goal is met by
  the slot-reorder semantics (not just the q+k relabel).
- kperm_branch_vs_m9_max = 0.125: NOT a wiring bug (parser label corrected). Depth-scaling
  signature (node2 0.0 / node4 3.1e-2 / node6 0.125 ≈ 2 ULP) = one-time lateral butterfly shift
  to the new FIXED layout; a wiring bug would corrupt all rows at O(1) (cf. KV-pad 16.3).
  Unbiased realization shift, stable in production (tree always cat8) — behavioral arbiters are
  the S2 gates (garble 0 + lossless-vs-nonspec).

NEXT: S2 e2e boot FR13_SLOT_REORDER=1 (eager first, then graph): engagement (runner pi log +
tree_attn bias pi log EQUAL + remap dst_pi live), garble gate, accept probe (expect cat8-spine
~3.5 vs prior ~3.18). Then S3 goal gate vs E5.

## S2a e2e + CONTROL A/B (2026-07-13/14, same harness, eager, cat8 B=1, captured SWE prompt)

ENGAGEMENT PASS: runner pi == tree_attn bias pi == [0,1,3,5,7,8,2,4,6]; "decode tree causal=False";
remap ENGAGED with foreign_first=0 on spine accept path0=[1,3,5] (= the predicted ZERO-COPY:
spine node depth d's permuted slot IS flat slot d). No crash, streams committed.

A/B (fix-ON s2a_eager vs flag-OFF s2a_control_off, identical config):
- Output class IDENTICAL (both emit turn-marker/near-empty text on this captured prompt —
  the S2a "garble" alarm was a BASELINE ARTIFACT, control does it too).
- brhist/event: spine 2.874 vs 2.784 (+0.090), branch 0.345 vs 0.338 (+0.006), total +0.096.
- DEEP-SPINE rows (the fix target): row5 0.504 vs 0.446, row7 0.454 vs 0.396, row8 0.412 vs
  0.360 — all +0.05-0.06 aligned = the deep-spine recovery signature.
- probes: temp06 3.655 vs 3.283 (+0.37); greedy 3.283 vs 3.322 (-0.04, noise).
CAVEAT (honest): n~130 events/arm, ONE prompt, cross-boot; per-row deltas ~1sigma each.
Suggestive, NOT proof. Arbiters: S2b garble gate (matrix_build temp06) + S3 matched-proof vs E5.

## COST + GENERALITY (2026-07-14, user questions)

**HBM tax / s-per-fwd: NONE by design, confirmed at greedy.** Per-step added work = two 9-elem
int64 gathers (permute+restore, metadata-side) + one [1,path] gather at commit; attention path
identical (same pages/bytes/FLOPs); causal=False REMOVES mask work; remap copies FEWER rows
(spine accepts zero-copy, foreign_first=0 observed live). Resident overhead ~1KB. EMPIRICAL
(eager, content-matched greedy pair): fix 0.1137 vs control 0.1150 s/fwd (-1%, noise). temp06
wall delta = trajectory-confounded (different outputs => host stream handling), not mechanical.
Deploy-grade timing = S3 graph mode.

**3-3-3 tree (challenge): GO, zero code changes.** pi is derived from SPEC_CONFIG at runtime —
logic model extended with choices_to_tree() (exact shipped algorithm, cross-checked against
served cat8): [3,3,3] => tree_n=10, spine [0,1,4,7] -> contiguous cols [0..3] == cat8's
canonical spine form (same M-invariant layout), 6 branches at fixed gapped cols, committer OK
at d1/d2/d3. Live 3-3-3 arm queued after S2b (SPEC_CONFIG + FR13_SLOT_REORDER=1, engagement
+ accept probe).

## S2b GARBLE GATE: PASS, non-vacuous (2026-07-14, s2b_garble_20260713T234233Z)

3 arms, 90 samples each, temp 0.6, identical prompts+seeds (fr13_garble_gate.py G1
undefined-name metric): native 0.00% (0/90) | treectrl cat8 fix-OFF 0.00% (0/90) |
**treefix cat8 FR13_SLOT_REORDER=1 0.00% (0/90)**. Zero syntax errors / empty gens anywhere.
Engagement audit: treefix logged BOTH pi lines (runner + tree_attn bias, equal, live during
generation); treectrl audit EMPTY (control genuinely off). => The slot-reorder fix is
GARBLE-CLEAN at the behavioral gate. S2 complete (S2a e2e+control, S2b garble).
NEXT: live 3-3-3 arm (running), then S3 goal gate (cat8+fix vs E5, graph, matched proof).

## S2c LIVE 3-3-3 ARM: PASS (2026-07-14, s2c_333_20260714T002751Z) — fix is tree-agnostic LIVE

Zero code changes. Engagement: runner pi == bias pi == [0,1,4,7,2,3,5,6,8,9] tree_n=10 (EXACTLY
the logic-model prediction); causal=False; remap engaged. STRUCTURAL PROOF: accepted spine path
[1,4,7] committed ZERO-COPY (foreign_first=0) — the permuted-slot identity (spine depth-d ==
flat slot d) held on a never-seen tree. Probes clean: greedy 2.446, temp06 2.534 accept/fwd
(structurally right for depth-3 spine; not comparable to cat8 depth-5). User challenge MET.

## S3 PROBE VERDICT — graph-clean + no-tax ESTABLISHED; probe is NOT the goal gate (2026-07-14)

S3 (s3_goalgate_20260714T003920Z, GRAPH mode, captured-prompt probe, 3 temps):
  cat8+fix: greedy 3.607 / temp06 3.145 / temp10 3.561; s/fwd 0.131-0.139; brhist spine/event
  3.106, branch/event 0.201. native E5: greedy 4.100 / temp06 3.727 / temp10 3.339; s/fwd 0.103-0.109.
ESTABLISHED: (1) FIRST GRAPH BOOT WITH FIX = CLEAN (engagement 2 lines, no capture crash —
pointer-stability design validated); (2) cat8's historical temp10 COLLAPSE (~2.92) is GONE (3.561);
(3) s/fwd cat8fix==cat8control (fix has no tax; the 0.131-vs-0.107 gap vs native = the known tree
overhead, NOT the fix).
NOT ESTABLISHED — and the probe CANNOT establish it: native greedy 4.100 is +0.6 above EVERY
historical E5 reading (3.34-3.53) because the captured prompt's content saturates the chain
drafter (probe content != live agentic work; the probe generated coherent easy text at greedy).
Per the binding-gate policy (live SWE-Verified agentic only; probes = diagnostic), the GOAL
VERDICT moves to the live B=4 16-task run: LAUNCHED slreorder_cat8_cache_b4 (cachefirst ship env
+ FR13_SLOT_REORDER=1, WALL=0), vs clean refs native+cache 3.050 accept / 8/16 resolve / 1
give-up + fix-on-remap cat8 ~3.3 accept (FR13_REMAP_SHIP_RESULTS.md).

## LIVE 3-ARM CAMPAIGN CRASH — PRE-EXISTING mixed-step staging bug, NOT the fix (2026-07-14 ~01:40)

Arm 1 (cat8+fix) died ~40min in: EngineDeadError "FR13_REPLAY_ROUTE: committer rows 3 != staged
spec decodes 4 for layer language_model.model.layers.0.linear_attn" (fail-loud assert, correct to
fire). Scheduler dump: 4 reqs x 9 tokens; only 3 in scheduled_spec_decode_tokens; the 4th =
CHUNKED-PREFILL TAIL of exactly 9 tokens (num_computed=28672, num_output=0) co-scheduled with the
3 tree-spec decodes. Mechanism: runner marks the prefill req num_decode_draft_tokens=-1 (correct,
3 spec); but the tree GDN STAGING block (patcher :5084-5125, flags[0].fill_(1) + flags[1].fill_(
num_spec_decodes) at :5116-5119) did NOT run that step, and flags[0] is NEVER zeroed (one-way
freshness, :5116 is the only [0] write) => the replay route consumed the PREVIOUS step's staging
(4) with fresh==1 => assert. The assert PREVENTED real corruption (stale bank replay).
NOT the slot-reorder: none of the 5 edits touch scheduling, GDN staging, num_spec_decodes, or
committer row publication; the trigger is data-dependent (prefill tail <= tree_n tokens
co-scheduled with spec decodes at B>1) and fires flag-ON or flag-OFF. Prior 16/16 runs never drew
this co-schedule. Fix design in flight: workflow wf_c3937df5-d49 (4 source-mappers + synthesizer)
— exact gating branch + minimal fail-loud-preserving fix (per-step staleness invalidation +
mixed-step staging or sound subset-replay; NO silent tolerance). Campaign relaunches after.

## MIXED-STEP CRASH FIXED: FR13_UNIFORM_DISPATCH_GUARD (2026-07-14)

Workflow wf_c3937df5-d49 (4 mappers + synthesizer) CORRECTED my mechanism twice: (1) flags[0] IS
cleared after every commit consume (:9093/:9168/:9865 — my grep missed the alias), so plain
staging-skip would fire the OTHER assert; (2) the crash class fired BEFORE (2026-07-07,
sl_cat8_cache_qc4, "committer rows 2 != staged 3" — pre-slot-reorder, cast-iron pre-existing).
TRUE MECHANISM: vLLM `_is_uniform_decode` is SHAPE-ONLY; a chunked-prefill tail of exactly
uniform_decode_query_len (9) tokens co-scheduled with spec decodes passes it => FULL bs=N
uniform-SPEC graph REPLAYS: captured staging fills re-stamp flags=[1,N] (capture-time constants)
AND the GDN builder's persistent-buffer refresh (gated num_prefills==0, gdn_attn.py:672) was
skipped => whole replayed forward consumed STALE spec_state_indices. Assert refused a genuinely
corrupt step (working as designed). SILENT SIBLING: pure prefill-tail batches (zero spec) replay
the spec graph with NO committer to catch it — plausible past B4 flake/garble carrier.
FIX (patcher _patch_gpu_model_runner_uniform_dispatch_guard, NOT flag-gated — pure correctness):
before _determine_batch_execution_and_padding, force_uniform_decode=False (existing stock kwarg;
None=stock) when num_spec_tokens>0 AND the step is not PURELY spec by the committer's own truth
(scheduled_spec_decode_tokens covering every req; no ndt<0). Demoted steps run PIECEWISE: mixed
metadata consumed, staging Python live (flags=[1,true_count]), committer rows match, prefill tail
takes the prefill path. num_spec_tokens>0 guard LOAD-BEARING (non-spec configs keep FULL graphs
on 1-token decodes). Host-side only; genuine uniform steps byte-identical (None). Also closes the
silent sibling. S0 PASS (17 patches sequence + markers). Rejected: per-step flag zeroing (replay
re-stamps after build), staging-gate widening (Python never runs under replay), subset-consume
(silent garble of stale rows — banned).

## LIVE ARM-1 FINAL: cat8+fix BEATS the native bar (2026-07-14, slreorder_cat8_cache_b4, 16/16)

**resolved 9/16 (56%)** vs native+cache ref 8/16, garble-era cat8 6/16 — BEST cat8 on this subset.
**give-ups 0/16** (every task wrote a patch; zero timeouts) vs native ref 1, garble-era 5.
**accept_per_event 3.500 clean** vs remap ref 3.336 (+0.16, single-flag delta) vs native 3.050.
deploy: s_per_fwd_gpu(draft) 0.0638, derived_tps_gpu 70.6 PROVISIONAL (prefill_frac 0.226 —
window-match vs native pending arm 2), sidecar 206.5 ms/step / 68.2 ms/draft (8832 steps).
Garble: trace-scan 3/16 "signatures" = heuristic noise (no near-neighbor identifier drift
anywhere; error-loops on 2 hard tasks = iteration-not-thrash; UNDEFINED hits are real API names
+ agent-authored dynamic attrs). Controlled S2b gate (0/90==native==control) stands; native arm
gets the matched trace-scan null. Crash-class 0 across the full arm (dispatch guard soak PASS).
Engagement non-vacuous (pi lines + causal=False; native arm 2 inverted-audit: 0 reorder lines ✓).
NEXT: native (running) -> cat6+fix (superset test) -> t33333. Speed research (task #25) launched.

## SPEED PROBE + SPEED-UP LIST (2026-07-14, wf_56f6537d; raw: FR13_TREE_SPEED_PROBE_AND_SPEEDUPS_RAW.txt)

MEASURED (matched 16-task windows, graph, cache-ON): tree cat8+fix wall 39.05 ms/committed-token vs
native 34.86 (+12%). Decomposition: verify-forward GPU/tok tree 13.36 < native 15.39 (tree CHEAPER —
accept 3.500 + 3.03 drafts/step amortize the weight read); derived_tps_gpu PARITY (70.6 vs 70.9);
THE ENTIRE deficit is NON-FORWARD: 25.69 vs 19.47 ms/tok (+6.2, +32%) = drafter/committer/gaps.
Slot-reorder already bought ~2.5 ms/tok (41.58→39.05) via accept 3.34→3.50. END-STATE BOUND: match
native's non-forward and tree WINS ~6% BEFORE accept gains. Sensitivity: −1 ms/tok = +2.6% TPS.

**S1 (TOP, 2-4d, est +2.5-8% TPS): the ship path runs the UNOPTIMIZED committer.** All EAGER_PACK/
OPT-1 collapses (DtoH 102→1, replay 96→2, packed writeback) are wired ONLY into the GREEDY twin
(_lumo_tree_path_lcp_max_greedy_sample); temp-0.6 serving dispatches _lumo_tree_canonical_
multidraft_sample which has NONE — ~75-110 blocking .item()/step + legacy 96-launch replay loop.
CORROBORATED INDEPENDENTLY by today's crash stack: rejection_sampler.py:3758 in the SAMPLED twin hit
the LEGACY per-layer flag check (:9815), not the _ep_stacks path (:9093). Port the collapses to the
sampled twin (greedy twin = working template); byte-identity + garble + same-boot A/B gates.
S2 (1-2d, +2-7%): Sequoia/OPT-Tree offline DP over the 9-row budget (reallocate, never remove
branches) — certify [6,6,4,6]/cat8 or emit better static shape. S3 (hrs+2-4d): measure drafter GPU
(wire DFWD timer into deploy report) then FR-Spec truncated draft vocab if lm_head read confirmed.
S4: re-test async_scheduling x spec compose. No-gos: per-forward compute opts (HBM floor), branch
removal (= deleting the accept lever).

## NATIVE ARM FINAL + RETRACTION (2026-07-14, native_ourcache_b4, same binary)

native: resolved 9/16, give-ups 0, accept_per_event 3.442, derived_tps 76.3 @ prefill_frac 0.157.
**RETRACTION: the matrix native+cache row (3.050, 8/16, 1 give-up) is STALE/pessimistic** — my
earlier "+0.45 accept over native" framing used it; the true SAME-BINARY delta is cat8+fix 3.500
vs 3.442 = **+0.058 (within temp-0.6 trajectory noise ±0.25)**. HONEST verdict so far: cat8+fix at
QUALITY PARITY with native (resolve 9/16 tie, 0 give-ups both, accept small-positive) + branch
rescue intact; the +12% wall/tok overhead (100% non-forward, S1-fixable) is what separates parity
from a win. The user goal axes: spine==native ACHIEVED (op-level bit-exact + live accept parity);
"branches save correctly" SUPPORTED (resolve tie, 0 give-ups, garble clean). derived_tps NOT
comparable across arms here (prefill_frac 0.226 vs 0.157 window mismatch — use the probe's matched
decomposition instead). cat6+fix arm running (superset test); t33333 after.

## SUPERSET CONFIRMED LIVE (2026-07-14, cat6+fix final): cat8 = cat6 + 0.166 (predicted +0.17)

cat6+fix final: resolved 9/16, give-ups 0, accept_per_event 3.333 (engagement tree_n=7
pi=[0,1,3,4,5,6,2] non-vacuous). THE VERDICT MATH:
  cat8+fix 3.500 − cat6+fix 3.333 = **+0.166 vs predicted ~+0.17** — the mechanism
  quantitatively confirmed: PRE-fix cat8 ≈ cat6 (spine M-dep −0.3 canceled the 2-extra-branch
  gain +0.17); POST-fix the cancellation is gone and the branches are PURE GAIN, to within
  0.004 of prediction. ALSO: cat6 3.333 < native 3.442 < cat8 3.500 — one branch does NOT beat
  the chain drafter; THREE branches put the tree over the native accept. Quality parity
  everywhere (9/16 + 0 give-ups on all three arms). Old cat6 "3.594" was trajectory noise
  (its own doc: structurally impossible); the MATCHED comparison is this one.
t33333 arm running (16 rows/req; ms_step 258.5 as expected heavier — its accept must clear
~+25% over cat8's to pay for the wider verify; the arm measures exactly that).

## S1 PORT DESIGN (2026-07-14, designed by direct source read — twins + kernel)

STRUCTURE: dispatch :10374-10445 (all_greedy -> greedy twin :7699; temp>0 tree -> sampled twin
:9259 via fr13_device_multidraft_commit). The sampled DECISION is already on-device (BAKED
FR13_DEVICE_MULTIDRAFT); the cost is the WALK TRANSPORT: device_multidraft_step
(fr13_device_multidraft_kernel.py:293-337) does 4-7 blocking .item()/NODE (overlap_mass :303,
source draw :311, token :313, u+accept :319, mass :330, residual draw :335) x ~10-15 nodes/step
x B=4 => the measured ~75-110 syncs/step (~6-10ms + pipeline bubbles).

**P1 (core): depth-synchronous walk, 100+ syncs -> ~5/step, BYTE-identical.** Rewrite the walk
so each DEPTH level runs all B requests' node-decisions as device ops WITHOUT .item()s (decision
tensors stay on device; next-node selection via device gather), with ONE tiny [B] "continue"
flag DtoH per depth (depth<=5 => <=5 syncs/step) + ONE packed products DtoH at the end.
CRITICAL CONTRACT: preserve the EXACT per-request draw sequence — same conditional draws, same
order, same per-request torch.Generator (loop over B for multinomial calls; B<=4, launches are
async ~5-10us vs 50-100us blocking syncs) => products BYTE-IDENTICAL to today's device path
(stronger than the distribution-lossless bar of fr13_device_multidraft_offline_gate.py, which
remains the fallback gate). REJECTED design: fixed-depth masked unroll with zero per-depth syncs
— it draws EXTRA (masked) rng samples, advancing per-request generators differently => trajectory
divergence (distribution-OK but breaks same-seed byte gates); take the 5 syncs.
**P2: transplant the _ep_stacks all-layer replay** (greedy :8990-9093 incl. flags consume) over
the sampled legacy per-layer loop (:9798-9865; ~96 launches + flag .item()s). Stacks are
module-level (_FR13_EAGER_PACK_STACKS on the gdn module, init-time) => directly usable; the
greedy block is the tested template (byte-A/B-gated at FIX-2).
**P3: G2.b-style packed writeback/publish twin** (greedy offset :8528+ vs sampled per-element
:9564-9787). **P4 (defer): batched conv-commit kernel (~400 launches, new kernel work).**
GATES per piece: offline distribution gate (P1) / byte-A/B (P2,P3) -> S0 patcher self-test ->
boot engagement + same-seed probe -> live same-boot A/B wall/tok. Flag FR13_COMMITTER_PACK_SAMPLED
default OFF until gated. Effort: P1 1-2d, P2 0.5d, P3 0.5d.

## S1/P1 IMPLEMENTED + BYTE-GATED (2026-07-14): FR13_DM_DEPTHSYNC

Depth-synchronous multidraft walk in fr13_device_multidraft_kernel.py (_fr13_commit_depthsync,
env FR13_DM_DEPTHSYNC default OFF; legacy path serves commit-trace diagnostics). Per LEVEL:
readback A (batched overlap masses) + readback B (source/accept) + readback C only on levels
with rejects + one batched seed sync + one final packed row DtoH => ~2 x walk-depth syncs/step
vs legacy 4-7 blocking .item() PER NODE (~100/step). Byte-identity engineering: single-row
softmax preserved (no [A,V] batching — 1 ULP risk), exact-k multinomial (replacement=False rng
consumption is size-dependent => padding banned), residual draws launched only for rejected
requests as their LAST draw, control compares replicate python-float semantics (u f32->f64 vs
accept_prob f64; mass floats via readback), residual divided by the DEVICE mass tensor.
GATE: scripts/fr13_dm_depthsync_byte_gate.py — **96/96 cases BYTE-IDENTICAL** (cat8/cat6root/
t33333 x B1/B4 x 4 seeds x random/dominant/adversarial/zero-overlap). NEXT: P2 (_ep_stacks
all-layer replay transplant into the sampled twin), then boot gates + live same-boot A/B.

## P1 B=4 A/B: walk-sync win NOT confirmed at ship batch; TIMERS REMAP THE OVERHEAD (2026-07-14)

B=4 live 4-task arms (engagement audited: dson ENGAGED=1/legacy=0, env pins correct, crash 0):
cfwd 44.1 vs 44.2 ms/span (NO delta); deploy committer_ms_per_step 53.3 vs 51.4 (-3.5%, but
prefill_frac mismatch 0.344 vs 0.293 => not clean); accept 3.60 vs 3.48 (noise). The B=1 probe's
-4.4ms/span is therefore INCONCLUSIVE (single boot pair; cfwd's dominant blocks — legacy replay
loop, conv commits, remap — carry cross-boot variance of the same magnitude). HONEST: the walk
.item()s were over-attributed by the census (est 3-8ms; real ≲2-4ms at agentic effective-batch
~1.3, where legacy sync count barely scales). FR13_DM_DEPTHSYNC stays DEFAULT OFF (byte-safe,
96/96-gated, value unproven — no bake).

**THE NEW OVERHEAD MAP (first-ever direct timers, both arms agree):**
  drafter (dfwd)   87.6-90.7 ms/step  <- THE ELEPHANT (~6.4 ms/tok at 13.6 tok/step; ~42% of a
                                          207ms verify forward; 5 sequential level-calls)
  committer (cfwd) 44.1-53.3 ms/step  <- replay legacy loop + conv commits + remap (P2/P3/P4
                                          targets), walk syncs minor
  verify forward   ~207 ms/step (sfwd)
=> The +6.2 ms/tok tree-vs-native non-forward gap decomposes as mostly DRAFTER + committer
blocks. RE-RANKED ATTACK: (A) drafter structure — is propose_tree graph-captured or 5 eager
host-orchestrated level forwards? measure + graph/fuse; FR-Spec vocab cut if lm_head-bound;
(B) P2 replay transplant (the 96-launch loop inside cfwd); (C) P3 writeback. Committer walk
(P1) done+shelved (flag exists if later useful at true B>2 concurrency).

## DRAFTER ATTACK, CYCLE 1 (2026-07-14): structure confirmed, hypotheses ranked

CONFIRMED (eagle.py:381-396): the drafter NEVER runs FULL cudagraphs — "Eagle only supports
PIECEWISE" — so each tree level is a host-orchestrated piecewise call (root topk + ~4 level
forwards + eagle_step_update_slot_mapping_and_metadata between levels). 88ms/step over ~4-5
calls = ~18-22ms/level for a TINY draft forward. HYPOTHESIS SPLIT (to sub-measure next):
(a) draft lm_head weight-read: [hidden 4k x vocab ~150k] bf16 ~1.2GB PER LEVEL on the HBM-bound
GB10 => ~6ms/level = ~30ms/step floor; (b) piecewise launch storm + per-level metadata/slot
rebuild host work = the rest. ATTACK CANDIDATES: (i) FR-Spec truncated draft vocab (cuts (a)
~5x; garble gate mandatory; committer q must match); (ii) level fusion / persistent draft graph
(cuts (b)); (iii) **S2 SHAPE TIE-IN — drafter cost scales with DEPTH (levels), NOT width**:
a shallower-wider tree cuts drafter cost linearly (~20ms/level) with zero kernel work — the
shape DP must now include ~20ms/level drafter term + accept-ceiling-vs-depth tradeoff
(t33333: depth-5, 11/16 resolve; a depth-3 wide shape would save ~40ms/step drafter).
NEXT CYCLE: sub-measure (a)-vs-(b) (one diagnostic boot: time lm_head vs rest inside a level,
or infer from a vocab-truncation dry-run), then design the winner. S2 DP design (mine) now has
its cost model: step_ms(rows) ~204+3.9/row(verify) + ~20ms x depth (drafter).

## S2 SHAPE DP RESULT + PARALLEL-DRAFTING CORRECTION (2026-07-14, hands-on)

**PARALLEL DRAFTING CONFIRMED by elimination** (split5/6 path-maps: propose_tree NEVER entered,
propose() tree-branch NEVER taken, chain-fallback NEVER taken; only the :483 early exit
`num_speculative_tokens==1 or parallel_drafting` remains) => the fork drafts ALL tree tokens in
ONE parallel pass. CORRECTIONS: (a) the "5 sequential level-calls" model was WRONG for this
config; (b) tree DEPTH does NOT buy drafter savings => that S2 lever is dead; (c) the 88-100
ms/step dfwd is ONE draft pass + sample — its internal split (model fwd vs lm_head/sample)
still needs the re-anchored split timer (split6/7 runs were externally stopped; measurement
pending GPU resume).

**S2 DP (scripts/fr13_tree_shape_dp.py, calibrated p_d/q_d from cat8 brhist; reproduces
measured ordering, ±0.1 on magnitudes — overestimates t33333 by +0.16):**
- Within TRUSTWORTHY space (depth <= 5): **cat8 [2,2,2,1,1] is near-optimal** — best variants
  ([2,2,2,2,1]) gain <= +0.07 E[len] for +1 row = wash. No cheap S2 win; shape stays cat8.
- Depth-6 shapes rank highest ONLY via unmeasured p6 extrapolation (0.915 from a small-n
  long-tail) + assumes the MTP-5 drafter can draft a quality 6th level — PARKED unless the
  drafter horizon is probed cheap.
=> Speed path narrows to: (A) drafter single-pass internals (split measurement then FR-Spec
lm_head cut or input-build fix), (B) P2 replay transplant (44-53ms committer span).

## SPEED CAMPAIGN — HONEST STRATEGIC STATE (2026-07-14)

CHEAP LEVERS EXHAUSTED. Measured, not asserted:
- P1 (committer walk depthsync): built, byte-gated 96/96, B=4 A/B NO win → shelved (flag OFF).
- S2 (tree shape): DP calibrated → cat8 near-optimal within depth<=5; depth lever DEAD (parallel
  drafting) → no shape win.
- Drafter internal split (model vs lm_head): timer built (propose-anchored) but 6 measurement
  boots inconclusive/interrupted (SIGKILL-vs-atexit, worker-env, external kills x2). UNMEASURED.
- P2/P3 (committer replay/writeback transplants): replay-loop SHARE of the 44-53ms committer span
  is UNMEASURED; transplant is correctness-critical + delicate. Not started (won't implement blind
  — the P1 lesson).

WHERE THE +12% wall/tok LIVES (measured): 100% non-forward = drafter single-pass (~88-100ms/step,
the biggest slice, HBM-bound weight reads) + committer (~44-53ms). GPU verify is at native parity;
cat8 accept 3.500 > native 3.442. So the tree is NOT compute-slower — the gap is host/HBM overhead.

REMAINING LEVERS ARE ALL EXPENSIVE + CORRECTNESS-CRITICAL:
  (A) FR-Spec truncated draft vocab — cut the ~1.2GB/pass lm_head read; multi-day; MANDATORY
      garble gate (committer q must match truncated draft dist); the only real drafter lever.
  (B) P2/P3 committer transplants — delicate; share unmeasured (needs 1 clean boot to justify).
Both need GPU boots to measure+validate; the box is currently contested (2 external kills).

RECOMMENDATION: this is an EARNED cost-gate. The garble+accept deliverable (SLOT_REORDER, superset
+0.166) is DONE and banked. Remaining speed is real (~6% end-state) but costs multi-day
correctness-critical work. Decision menu for the user:
  1. Measure-first: ONE clean boot with the drafter split timer (built) + a committer replay
     sub-timer (to build) → decides A-vs-B before spending days. Cheapest next step.
  2. Commit to FR-Spec (A) — highest ceiling, hardest, garble-gated.
  3. Stop speed here — ship the correctness win; the tree is at accept+GPU parity, host-overhead
     slower, which for an HBM-bound agentic workload is a defensible ship.

## DRAFTER DECOMPOSED (2026-07-14, split10 detached): HOST-BOUND, FR-Spec DEAD

Precise GPU-event spans on the PATCHER's OWN drafter (not pristine — the fork rewrites propose via
FR13_DRAFTER_SINGLE_LOGITS): model(draft fwd) 8.3ms + lmhead(compute_logits :13361) 15.1ms = 23ms
GPU COMPUTE per step; dfwd TOTAL 140ms/step. => ~117ms (83%) is HOST orchestration + GPU-idle gaps
(CUDA events count host-idle between records). The DRAFTER IS HOST-BOUND, not HBM/weight-read-bound.
- FR-Spec (truncated draft vocab) REFUTED as a lever: lm_head is 15/140; halving saves ~2% total.
  Do NOT pursue.
- model forward (8.3ms) confirmed NOT the bottleneck (earlier split9 5.5ms consistent).
- The 117ms host = prepare_inputs (draft input build from accepted path) + tree bookkeeping +
  per-step host tensor ops + .item() syncs, in propose_draft_token_ids. This is the real drafter
  lever: reduce host Python / syncs / graph-capture prepare_inputs. DIFFUSE + needs host profiling.

INFRA WIN: fr13_detached_boot.sh (setsid session-detach) DEFEATS the phantom boot-killer — split9
+ split10 both completed (death.log clean, no signal) after split6/7/8 died at ~50s-3min. Root:
serve script `trap teardown EXIT` binds container life to launcher; a SIGTERM to the (background)
launcher tripped teardown -> docker rm -f. Detaching into a new session breaks the chain.

ATTACK re-ranked (all cheap/compute levers now refuted — P1 walk, S2 shape, drafter model, FR-Spec):
  (A') drafter HOST overhead (117ms) — biggest, but diffuse host Python; needs profiling.
  (B) P2 committer replay transplant (bounded, template-proven) — share of 44-53ms cfwd unmeasured.

## SPEED CAMPAIGN CONCLUSION — EARNED COST-GATE (2026-07-14, fully measured)

Every CHEAP lever measured or measured-analogy refuted. The tree's +12% wall/tok is STRUCTURAL:
| lever | verdict | basis |
|---|---|---|
| P1 committer walk (depthsync) | no B=4 win | DIRECT measure (byte-gated, A/B) |
| S2 tree shape | cat8 near-optimal | hands-on DP on 4 measured arms |
| drafter model forward 8.3ms | not bottleneck | split10 GPU span |
| FR-Spec / lm_head 15ms | REFUTED (11% of drafter) | split10 GPU span |
| drafter host 117ms/140 | THE elephant, STRUCTURAL | split10 (23ms GPU / 117ms host); eagle PIECEWISE-only (eagle.py:384) |
| P2/P3 committer collapses | inferred-marginal (NOT direct) | same sync/launch-collapse class as P1 (no-win); committer 44ms is mostly real recurrent-replay COMPUTE, not launch overhead |

REMAINING REAL LEVER = make the eagle drafter FULL-cudagraph-capturable (kill the 117ms host
orchestration between piecewise pieces). This is a deep vLLM-upstream-limited change (weeks), for
~6% on a tree already at GPU-parity + accept 3.500>native 3.442. NO plausibly-cheap correct path
remains (per feedback_speed_is_the_goal_cost_gate) — MEASURED, not assumed.

SHIPPED + BANKED: FR13_SLOT_REORDER (superset +0.166, garble-clean, accept>native). Infra: GPU span
timers (sfwd/dfwd/cfwd + dfwd 3-way split), fr13_detached_boot.sh (defeats the trap-teardown killer),
fr13_tree_shape_dp.py. HONEST caveat: P2/P3 are inferred not directly measured — if speed is refunded,
measure the committer replay share first (delicate committer edit) before the structural drafter work.

## FRONT 1: "IS 16 THE LIMIT?" — YES but SOFT/SELF-IMPOSED (2026-07-14, tree_scale sweep)

B=1 sweep (num_tokens=tree_n, no batch-tile confound): cat8(9)=131.3ms, t33333(16)=145.0ms
(SMOOTH ~2ms/row, NO cliff; accept RISES 3.691->3.962 with tree size). t55555(26)=BOOT FAIL.
ROOT: `NotImplementedError: FR10 GDN tree verifier only warms padded tree sizes <=16, got 26`
(gdn_attn.py:294 = patcher :236): n_pad=1<<(n-1).bit_length(); if n_pad>16 raise. So tree_n<=16
=> pad16 OK; 17-32 => pad32 => raises. **The 16 ceiling is the GDN verifier warmup-buffer
padding sized for pad-16 — NOT a kernel/tile/graph physics wall.** Corroborating: FA2 query
tile=64 rows; cudagraph_capture_sizes=[1,2,4,8,16,24,32,40,48] max48 (16 not special). Raising to
pad32 = bounded change (widen the >16 guard + the strict/visible [n_pad×n_pad] masks + replay
ring/state buffers to 32). VERDICT: verify scales linearly (~2ms/row, weight-read-dominated),
accept keeps rising with width; 16 is a soft cap, pad-32 trees are feasible at B<=1 (graph-captured
<=48 tok) but B=4 past ~12 rows/req falls to eager unless cudagraph_capture_sizes widened. Whether
to go past 16 = the width-economics (bigger accept vs ~2ms/row + drafter host + graph ceiling).

### FRONT 1 addendum: does >16 nodes SPILL SRAM->HBM? NO (kernel-read, 2026-07-14)
User hypothesis: trees >16 nodes spill SRAM to HBM (=> raising the cap is NOT cheap). CHECKED
against the verify kernel /tmp/vllm_fla_fused_sigmoid_gating.py (fused_sigmoid_gating_delta_rule
_update_kernel): the GDN tree verify is a SEQUENTIAL RANK-1 recurrence -- state tile
`b_h = tl.zeros([BV, BK])` (line 106) sized by HEAD dims (BK=next_pow2(head_k), BV=min(next_pow2
(head_v),32), line 223), and `for i_t in range(0,T)` (line 136) STREAMS the T=n_pad tree nodes one
at a time. SRAM footprint is head-dim-sized, TREE-SIZE-INDEPENDENT; 16->32 nodes = 2x loop trips,
same SRAM. No spill. (Kernel sig doesn't even take the [n_pad x n_pad] masks -- ancestry via
per-node h0 initial-state indices.) The "16=spill" memory most likely conflates: (a) the CHUNKED/WY
delta-rule (PREFILL only, chunk_gated_delta_rule :5834) which DOES hold an intra-chunk [BT x BT]
SRAM matrix with BT a 16/32/64 tile; or (b) FA2 full-attn verify whose query tile = kNWarps*16 = 64
rows (tree fits one SRAM tile until 64 nodes, not 16). VERDICT UNCHANGED and STRENGTHENED: 16 is the
warmup-pad guard, not an SRAM wall; raising to 32 adds only the measured linear ~2ms/row (loop trips
+ FA2 query rows <=64), NO SRAM-spill HBM tax.

### FRONT 1 CORRECTION (user was RIGHT — 16 IS a hardware/BV register wall, 2026-07-14)
RETRACTS the two prior Front-1 blocks' "soft cap, cheap to raise" verdict. I read the WRONG kernel
first (stock fused_sigmoid_gating recurrent). The ACTUAL tree verify = launch_tree_gdn_prepared in
lumo_flywheel_serving/fr10_gdn_tree_kernel.py, whose hot kernel holds:
  h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K), fp32)   # line 408, BLOCK_V=BV=16, DIM_K=128
a REGISTER-RESIDENT cache of ALL N_PAD node states at once (line 405 comment: children start from
parent's fp32 checkpoint "without reloading h0 or replaying ancestors from HBM" -- the design's
speed trick). Footprint: N_PAD=16 -> 16*16*128 = 32768 fp32 = 128 KB/CTA = HALF the SM's 256 KB
register file, JUST for h_cache. N_PAD=32 -> 256 KB = the WHOLE file -> guaranteed spill to local
memory (HBM). **This IS the SRAM->HBM spill; BV=16 is the middle dim driving it.** So 16 is a real
register-capacity boundary, NOT a warmup convenience.
MEASUREMENT ERROR I MADE: my "smooth 9->16 rows ~2ms/row" sweep ran cat8(9 nodes) AND t33333(16
nodes) BOTH at n_pad=16 (padded_nodes rounds up to 16); the 131->145ms delta was more UNMASKED nodes
inside the SAME 16-wide tile, NOT crossing 16->32. I never measured n_pad=32 (guard raises first).
Zero data past the boundary => the "linear/cheap" extrapolation was unfounded.
CORRECTED VERDICT: raising past 16 is NOT free -> h_cache spills to HBM. To widen you must RE-TILE
the kernel: e.g. BLOCK_V 16->8 (N_PAD=32 -> 32*8*128 = 128 KB, back under ceiling) at 2x V-grid
programs + redundant q/k reloads -- a kernel rewrite with real overhead, not a constant bump.
IMPLICATION for the drafter interface: "free wide branches" is NOT free at the verify kernel past
n_pad=16 -- the register wall gates tree width; wider merged trees need the BLOCK_V re-tile first.

### FRONT 1 red-team CLOSED (2026-07-14): failure is at INIT, not a graph/tile confound
t55555(26 rows) rc=2 with `NotImplementedError: FR10 GDN tree verifier only warms padded tree
sizes <=16, got 26` logged at EngineCore START (07-14 18:15:54, core.py:1129) — i.e. at
initialize_kv_cache/create_metadata_builders, BEFORE any decode step or cudagraph capture. This
REFUTES the "it's really the 64-token FA2-tile/cudagraph boundary not tree_16" alternative: the
hard cap bites at n_pad=32 at init, well below the kNWarps*16=64-row FA2 boundary. The 64-token
boundary is a SEPARATE higher ceiling (would only matter at B=4 where tree_16 = 64 tokens — the
batch×tree confound the B=1 sweep was built to isolate, and did). Sweep curve (n_pad=16 for BOTH):
cat8(9)=131.3ms/3.691, t33333(16)=145.0ms/3.962, t55555(26)=BOOT-FAIL. FRONT 1 CONCLUDED: 16 is a
real init-time register-wall cap; widening = BLOCK_V re-tile (kernel rewrite), tracked but not cheap.

## FRONT 2 GATE 1 PASSED: suffix candidates committer-transparent (CPU proof, 2026-07-14)
scripts/fr13_suffix_committer_contract_gate.py — 32/32 vs the REAL deployment rule
(host_multidraft_accept_probs = draft_probs=None MTP path). Proven analytically + Monte-Carlo:
(P1) OUTPUT is EXACTLY the target p for ANY candidate set (P(out=t)=min(q_mix,p)+max(p-q_mix,0)=p);
adding suffix/garbage NEVER changes the output => LOSSLESS, UNCONDITIONAL. (P2) ACCEPT RATE = p(S)
= target mass on the DISTINCT candidate set; adding a p>0 candidate RAISES it (monotone; measured
gains +0.03..+0.20 for a few high-p suffix tokens). => suffix = a PURE MONOTONE ACCEPT LEVER.
(P3) garble-safety: a garbage token g (p~0) is committed at rate EXACTLY p[g]; the safety is the
SOURCE-SELECTION weight (=p[g]/overlap_mass ~0), NOT accept_prob (which = overlap_mass, not small)
-- a subtle point the gate CAUGHT (my first assertion "accept_prob~0" was WRONG). (P4) deterministic.
RED-TEAM: 4 initial FAILs were MY incorrect assertions, not code -- fixing them to the correct math
made the claim STRONGER (not "distribution unchanged" -- output invariant + accept monotone). This
de-risks the live seam BEFORE writing it: the merge is provably lossless + a speed lever. MergedSource
MUST dedup candidates for the clean accept=p(S) identity (lossless holds even without dedup).
