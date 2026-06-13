# FR13 substitution localization — v1 INVALID (no positive control), v2 corrected

Date 2026-06-13. Monitor red-team caught a broken instrument BEFORE it could produce a
false "diffuse" verdict. HEAD 811a4fe4. v1 workflow `wf_af5878a4-ec6` (wfgjalb96) KILLED
mid-run; v2 `wf_c1bcc077-c1e` (wm1xdgw7b) launched.

## What v1 did and why it's INVALID
Goal: substitution-localize the cat9 pos21 flip (prompt2: served **1970 `' code'`** vs
clean no-spec oracle **3425 `' files'`**). Method: splice the oracle's residual into the
tree verify-forward at a row, after a target layer (FR13_HIDDEN_SUBSTITUTE,
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:11771-11804` — overwrites
`hidden_states[frow]`+`residual[frow]`), measure if pos21 reverts to 3425.

**The bug (class 9 engagement, the whole point of the test):** every v1 arm used a
**hardcoded** `forward_row:6`, `oracle_row:0`, `skip:6` — none derived from pos21's commit
provenance — and there was **no positive control**. The smoking gun:

| arm | layers spliced | pos21 result |
|---|---|---|
| arm1 (baseline, disabled) | — | 1970 ` code` |
| arm6 | [0] | 1970 |
| arm3 | [58] | 1970 |
| arm11/arm2 | [63] | 1970 |
| arm12 | [62,63] | 1970 |
| **arm10** | **[0..63] ALL 64 layers** | **1970** |

Splicing the oracle's full residual at **all 64 layers** left pos21 at the flip value.
If `frow` were pos21's committed row and `oracle_row` predicted pos21, splicing the
**final** hidden would force lm_head to the clean argmax 3425. It didn't ⇒ **the splice
never touched pos21's computation.** The "no revert anywhere" is an instrument artifact,
**not** evidence of diffuse accumulation. v1 had no positive control to catch this and
would have falsely concluded "diffuse." (cat9 row layout: root@0, spine@1-5, branches
`(0,1)/(0,0,1)/(0,0,0,1)/(0,0,0,0,1)`@6-9 — so `forward_row:6` = a depth-2 *branch* node,
almost certainly not pos21's committed row.)

Two concrete defects: (a) `forward_row` guessed (must be pos21's committed flat-row,
possibly permuted by the `_fr12_tree_candidate_pre/post_remap` tree-remap); (b)
`oracle_row:0` wrong (must be the row that **predicts** pos21 — the last decoded position
of the no-spec sequential decode; the hook's own default is `o_rows[-1]`, v1 overrode to 0).

## v2 correction — gate the bisection on a POSITIVE CONTROL
1. Derive `(skip,frow)` from **commit provenance** (forward index that commits pos21 +
   the accepted node's flat-row), mapped through the remap — or **search** `frow∈0..9`
   validated by the positive control.
2. `oracle_row` = the row predicting pos21 (no-spec sequential decode of prefix [0..20];
   confirm its lm_head argmax == 3425).
3. **MANDATORY GATE:** first establish a splice that actually reverts pos21→3425 at layer
   63 (with `FR13_HSUB ARMED`+`spliced layer 63 row R` warnings in docker logs). If no
   `(skip,frow)` reverts pos21 even at the final layer ⇒ FAIL LOUD, no bisection.
4. Only then layer-bisect to answer localizable-layer vs carrier-band vs diffuse.

## Red-team criterion banked (reusable)
For ANY causal-substitution localization: **before trusting a "no revert ⇒ diffuse"
conclusion, require a positive control** — a splice (typically oracle@final-layer at the
target row) that demonstrably FLIPS the measured output to the clean value. Absent that,
"no revert" means "the splice didn't fire on the measured unit," not "the divergence is
diffuse." This is the same instrument-confound class as node7 (wrong topology) and the
prefill-reference (wrong oracle) — localization keeps dying on the
served-position→(forward,row,oracle-row) mapping. Pairs with
[[feedback_fail_loud_assert_engagement]], [[reference_scalar_metric_per_token_blindspot]].
