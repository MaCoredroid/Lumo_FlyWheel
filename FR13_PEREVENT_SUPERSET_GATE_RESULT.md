# FR13 — Per-event superset gate RESULT: cat9 PASSES (lossless superset of E5), and the "23 flips" decompose

Date 2026-06-15. Workflow `w02jpqib2`, **verify HOLDS=True** (rigorous: verifier corrupted a stream to confirm
JOIN fail-loud, recomputed all numbers, confirmed spine_regressions=0 over 250 dump records). Reducer committed
77e2a0e8 (scripts/fr13_perevent_superset_gate.py). The user's precise per-event gate (replaces the cross-
trajectory aggregate 3.198 vs 3.076).

## The gate (banked fork-margin boot: 118 events, 466 positions, 4 prompts, greedy)
| | spine_reg | gross saves | lossless | lossy | net |
|---|---|---|---|---|---|
| d0 | 0 | 0 | 0 | 0 | 0 (dead depth-1 sibling — no d0 gain) |
| d1 | 0 | 11 | 7 | 4 | +3 |
| d2 | 0 | 9 | 7 | 2 | +5 |
| d3 | 0 | 3 | 3 | 0 | +3 |
| d4 | 0 | 4 | 4 | 0 | +4 |
| **TOTAL** | **0** | **27** | **21** | **6** | **+15** |

**GATE PASS: net = 21 − 6 − 0 = +15 > 0 AND spine_regressions == 0.** By the precise per-event gate, **cat9 IS
a lossless superset of E5**: it never serves fewer spine tokens (0 regressions, structural — committer strict
`>best_lcp` spine-favored tie-break, confirmed live + over all 250 dump recs, delta dist {0:193, 1:57}), and
its leaf gains are NET lossless. **Lossless fraction = 21/27 = 78%** of the leaf-save speed gain is genuine
lossless superset; ~22% (6 saves) is lossy flips (deviation 1.125–9.0 nat). net_lossless/event = +0.127.

## The decomposition that reframes "cat9 = 23 flips vs native 3"
The 23 clear-margin flips split (independently confirmed 6+5+12+0 = 23):
- **6 = lossy LEAF-SAVES** (leaf-fork flips on a leaf-save position) = the leaves' lossy cost. Net of the 27
  saves they are MORE than paid for (+15 net lossless).
- **17 = SPINE-realization / bonus flips** (5 spine-accept + 12 bonus, NOT leaf-saves) = cat9's tree-verify
  SPINE drifting from the decode oracle (the diffuse GDN/full-attn realization). **This is the bulk of the gap,
  and it is NOT the leaves.** native E5's own spine has ~3 such; cat9's tree-verify spine (forked-FA2 + tree-
  scan, vs native FLASH) drifts more → ~17.

So: **the LEAVES are a net-positive lossless superset gain (+15); the remaining gap is cat9's tree-verify
SPINE realization vs decode (17), the same KIND native has but more (tree-verify != FLASH spine).** The
"23 vs 3" was conflating the 6 leaf-cost with the 17 spine-drift.

## Two distinct gates, two answers (honest)
- **Per-event SUPERSET gate (the user's deliverable: cat9 >= E5 + net-lossless leaves): PASS** (+15, 0 reg).
- **Absolute lossless (cat9 flips <= native floor): NOT met on this boot** (cat9 23 vs native 3) — because
  cat9's tree-verify spine drifts ~17 more than native's FLASH spine; the leaves are not the cause.

## Caveats + next
SMALL-SAMPLE (#12): one boot, 23 flips / 27 saves / 118 events / 4 prompts; fork_margin OFF set per_prompt
[4,6,7,6] (different boot than scan_align [5,4,5,9]). TIGHTEN with the big-denom serve (if it co-arms the dump)
or a fresh paired dump boot. Greedy only (rescue real, 86a255a4). Non-vacuity all green (RECURRENT oracle
44352 calls, JOIN fail-loud 4/4 coverage 1.0, int-view not atol, det). NO bake/ship (user call). The big-denom
SWE-quality gate answers whether the residual 17+6 drift changes the task OUTCOME; the isolated-fork test
(queued) splits the leaf lossy-saves co-residency-vs-irreducible. Links [[reference_scalar_metric_per_token_blindspot]],
[[feedback_depth_matched_accept_compare]].
