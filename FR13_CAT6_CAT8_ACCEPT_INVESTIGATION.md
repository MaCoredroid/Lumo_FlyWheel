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

## RESULT
(pending — runs after native arm completes; recorded here.)
