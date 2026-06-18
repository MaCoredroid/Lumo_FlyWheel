# FR13_RESHAPE_WIDE — session closeout (wide-sweep: does wider-not-deeper help?)

Goal: build a general width-N tree drafter and measure whether *wider* (more candidate
leaves) at fixed/shallow depth helps deploy speed vs depth-matched native MTP, B=1 temp-0.6
on the codex+SWE-Verified deploy (subset_b4_four.json). **Answer: NO — and at zero
losslessness cost.**

## What was built
- **General width-N caterpillar drafter** (`FR13_RESHAPE_WIDE`, in fr10_phase4_patch_vllm_tree_gdn.py):
  reads `topk(width)` off each spine node and packs rank-1..K-1 runner-up leaves in sorted
  tree order, derived entirely from `tree_choices` — supports top-5/10/20/any width. Additive,
  exact-shape-guarded (cat9/333/cat6root/… byte-identical). Audited width-5-safe across the
  host+device committer / ancestry masks / conv-window / eager-pack (all fanout-general).
- **Shapes**: cat555 [5,5,5] (15-node depth-3), cat55221 [5,5,2,2,1] (15-node depth-5). Note the
  GDN tree-verifier warm cap is `n_pad = next_pow2(nodes+1) <= 16` ⇒ **max 15 nodes** (cat55222
  [5,5,2,2,2]=16 overflowed to pad-32 and was reshaped to cat55221).
- **Drafter-forward GPU timer** `FR13_DFWD_GPU_TIMER` + committer `FR13_CFWD_GPU_TIMER`
  (default-OFF span timers mirroring the verify timer; drafter wired around propose) for the
  per-length TPS-attribution instrument. + `scripts/fr13_tps_attribution.py` (formula reducer).

## Verdict — wider-not-deeper does NOT help (clean-task, depth-matched)
| depth-5 | nodes | derived_tps_gpu vs E5 |
|---|---|---|
| cat6root | 6 | **+15.7%** (and +27% deploy-WALL) ← winner |
| cat9 | 9 | +3.3% |
| cat55221 (wide) | 15 | −0.4% (tied) |
| depth-3: cat555 (wide) vs E3 | 15 | −3.3% |
The tps_gpu curve peaks at the moderate **cat6** and erodes with more width (verify tax grows
faster than accept gain). Wide trees tie/slightly-lose to native and lose to cat6. Deployed
**cat6 stays the best shape**.

## Lossless gate — BOTH wide arms PASS (temp-0.6, per-token clear-margin vs no-spec recurrent oracle)
- cat555 (d3 wide) 10.74% vs E3 floor 10.83%, z=−0.19 → **LOSSLESS** (indistinguishable).
- cat55221 (d5 wide) 10.96% vs E5 floor 12.90%, z=−3.52 → **LOSSLESS** (below floor).
- engaged+det valid (non-vacuous). The drafter-agnostic committer holds for wide trees.

## Key methodology lessons (both user-caught)
1. **Aggregate accept/tps is trajectory-confounded.** Two retry-heavy tasks (astropy-13236/13398)
   had the codex agent give up + retry nondeterministically ⇒ diverged trajectories that deflated
   cat55221's accept and produced spurious −14%/−8% verdicts. The CLEAN tasks (12907, 13033, same
   lossless trajectory) are the apples-to-apples basis. Per-task accept confirmed the wide drafter
   is a CORRECT superset (accepts the most on clean tasks) — no bug.
2. **+27% vs +15.7% are different metrics, not a contradiction.** +27% = `derived_tps` (deploy
   WALL basis, full per-step cost where higher accept stuffs more tokens into a ~fixed-wall step);
   +15.7% = `derived_tps_gpu` (verify-GPU-only, prefill-independent — the fair cross-arm basis,
   since WALL is prefill-confounded across arms, e.g. cat55221's WALL gave a garbage +133%).

## Research (committed)
- **Prefix caching for GDN-hybrid** (prefix_cache_gdn_hybrid_research.md, VERDICT YELLOW): off is a
  soft default (mamba_cache_mode='none'); the enable path (`--enable-prefix-caching`→align mode) is
  already in our build (#30877/#33705), but open upstream APC+MTP-spec regression #43559 = our exact
  combo ⇒ measurement-gated. Big prefill lever (short turns are prefill-bound).
- **FR9 node-count reconcile** (fr9_nodecount_scaling_reconcile.md): "3→5 nodes ~free" does NOT
  contradict FR9 (verify is weight-load-bound/row-flat) but was misleading (depth cost lives in the
  DRAFTER, not verify) and the +1.7% was within the cross-boot/cross-trajectory noise floor;
  RETRACTED the +0.0015 s/node model. Drafter forward = the small draft head + lm-head, NOT a full
  27B weight-load.

## Parked (await user)
1. Forward-cost campaign for the drafter/committer TPS-attribution instrument (timers built, OFF).
2. Prefix-cache enable + lossless measurement (YELLOW, cheap, GPU-gated).
3. cat6 is the winner — any further tuning is on cat6, not the wide trees.

Branch fr13-speedfix → merged to main. Default serving path unchanged (all new behavior
flag-gated OFF or exact-shape-guarded). GPU idle.
