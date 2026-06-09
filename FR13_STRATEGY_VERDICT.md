# FR13 STRATEGY VERDICT (workflow wiyva0b3s, adversarial, repo+literature-verified, 2026-06-09)

## TL;DR: we've been grinding LITERAL-0.0 per-layer when the deliverable gate is per-depth ARGMAX + within-E5-floor. There is a BUILT+MEASURED lossless +6.85% path (multi-spine) we've ignored. Bit-exactness buys ~0 TPS on a bandwidth-bound decode.

## 1. Tension = ENGINEERING, not fundamental
No impossibility theorem (Traversal Verification 2505.12398 Thm 3.3, SpecInfer Thm 4.2, Multi-Draft 2410.18234 Thm 1 = distributional losslessness regardless of topology). One irreducible floor: fp reduction-order non-associativity (sequential vs chunked traverse different reduction trees - "Defeating Nondeterminism", Thinking Machines) = why WY can't be bit-exact + the FA2 2-ULP floor. BUT that floor (0.0039 max) is ~15x BELOW the deliverable's bag-TV 0.0593 gate -> does NOT bind.

## 2. The bar mismatch (VERIFIED in our own docs)
- FR13_FR19_HANDOFF.md:21 = "bit-exact-or-bust, all-layers-0.0" (what the monitor drove).
- FR13_FLOOR_WORKFLOW_VERDICT.md:37 + FR13_SEQ_E2E_ROADMAP.md = "within-E5-floor / argmax-lossless (NOT literal-0.0); GATE = per-depth ARGMAX 4/4 + bag-TV within floor".
- Nuance (FR13_FLOOR_WORKFLOW_VERDICT.md:34): GDN spine needs drift=0 in the ARGMAX sense (drift=0 ⟹ superset by math; a drifting spine breaks superset) - but NOT literal-0.0 max_abs (within-floor max_abs is fine if argmax matches). The L4 0.0126 MAY be within-floor (no argmax flip) - unmeasured.

## 3. Better path = multi-spine (BUILT + MEASURED, VERIFIED)
fr9-swap-mtp-spine-tuning-plan-20260601.md:70-73 + fr9-superset-closeout:21-29: spines=2 winner 3.442 accept / 47.50 tok/s vs spines=1 44.45 = +6.85% TPS, superset_violations=0 / missing_sum=0 / 19,307 events, spine-0 ≡ E5. Lossless-by-construction, native MTP chain, FLASH_ATTN, NO custom kernel. The ONLY positive TPS number measured. The sequential tree-scan has ZERO e2e numbers. Literature (Snakes-and-Ladders NeurIPS'24) = SOTA exact recurrent backtracking is chain/batch NOT token-tree -> no-copy GDN tree is genuinely hard, not a missed easy method.

## 4. Premature death: RETRACT the FR10/FR11 "diffuse drift = unfixable" CONCLUSION
The sequential ladder decomposed the "diffuse" drift into named fixable roots (scan e4a6a2f2; the rest) + drove L0-3 to literal 0.0 - further than the replay kernel ever reached. Per feedback_math_correct_vs_bitexact: "'diffuse' often means 'never insisted on zero'." The replay KERNEL FORM stays dead (O(N²) HBM tax).

## 5. The insight we're missing
accept/event has a hard theoretical ceiling (Multi-Draft 2410.18234; OPT-Tree 2406.17276) native MTP-5 (3.076) nearly saturates. A lossless tree's upside over native is a THIN slice; bit-exactness contributes NOTHING to it. The speed lever is TREE SHAPE / DRAFTER QUALITY - which multi-spine's +6.85% already captured. If the lossless tree can't clear 3.076 by more than the register-resident replay tax, multi-spine's +6.85% is the best banked number, full stop.

## RECOMMENDATION (workflow): e2e-FIRST with the spine-argmax guard, at the within-floor bar.
STOP grinding per-layer literal-0.0. Measure the sequential tree-scan e2e NOW: per-depth spine ARGMAX match (necessary, attributable) + bag-TV ≤ 0.0593 + accept/event ≥ 3.076 + TPS. If spine argmax matches every depth + bag-TV ≤ floor -> DONE regardless of L4 max_abs. If argmax flips -> THEN localize. Don't pivot off the sequential tree-scan (research endgame); pivot off the literal-0.0 methodology. Keep multi-spine +6.85% as the banked floor to beat.
