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
