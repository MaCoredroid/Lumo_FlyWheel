# FR13 FIX-A Gate Bind — FR13_TREE_SAMPLE_ROW PASSES ALL GATES; FINGERPRINT NORMALIZED TO CHAIN-CLASS

Date: 2026-06-12 UTC. Repo @ HEAD df1dfa07 (branch main; FIX-A1 patcher/launcher
edits + tests + this bind left UNCOMMITTED for the monitor). Spec =
FR13_CHASE_STEP1_BIND.md "Next step — FIX-A" / wf_a71e2a24 nextStepSpec.
Implementation = the FIX-A1 + instrument-debt working-tree report (FIX-A1
injection at `_patch_eagle_tree_consumption_verify`, committer NROWS freshness
publish x2, REQKEY pre-forward clear, instrument-(iv) bare-tensor locator
repair + fail-loud, H3 minimal probe, FAIL-2 'rows' restore; 31/31 wiring
tests, 445-suite pass, live-source anchors applied=True).

## Verdict

**ALL HARD GATES PASS** (G1-G6 below) and the H1-ROWBUG fingerprint
**NORMALIZES to chain-class**: `h1_row_mismatch` 84/84 (step-1 stock) ->
**0/232** under the fix (in-process dual-path selfcheck, BY CONSTRUCTION:
sampled row == published leaf row on every decode event); ROWBUG-class
transition root-match 0.418 -> **0.90-0.949**; root-reject 39.7% ->
**12.4-13.8%** (chain-class 10-15%); next-spine-LCP after L>=3 accepts
0.97-1.53 -> **3.04-3.26**. Chain5 is regression-clean (accept in band,
within-boot byte-identity, cross-boot forks floor-class). Accept/event draw
(clean captured, never the gate): cat9 greedy **2.0274 -> 3.1789**.
`FR13_TREE_SAMPLE_ROW` stays **default OFF** in the launcher — the
default-flip / close decision goes to the user per standing policy.

## Campaign (6 boots, serialized, launcher-only; artifacts `output/fr13_chase_fixA/`)

Regime: canonical FIX2/FIX3-bind regime — PORT 9950, GPU_UTIL 0.82,
MAX_NUM_SEQS=1, BATCH_INVARIANT=0, FR13_BI_TREE_ATTN=0, FR10_METRICS=0,
FR13_REPLAY_ROUTE=1, FIX-1/2/3 committed defaults on every arm; pinned
prompts `output/fr13_acceptance_ladder/prompts_swe4.json`, seed 1313, B=1,
greedy + t0.6 x2 reps, 128 tok (warmup 1x16); docker rm -f between arms;
per-probe window snapshots. THE ONLY VARIABLE = `FR13_TREE_SAMPLE_ROW`.
Runner `output/fr13_chase_fixA/run_fixA_arm.sh`; reducer
`output/fr13_chase_fixA/reduce_fixA.py` -> `fixA_reduce.json`. Reducer
machinery validated against the banked step-1 boots first (reproduces
2.0274 accept, 39.7% root-reject, 0.418/0.800/0.904 fingerprint, hist
58/13/15/17/13/30 cell-exact). Campaign wall 20:27:30Z -> 21:25:49Z.

| arm | tsr | mode | healthy | needles (class 9, both-states) | probes |
|---|---|---|---|---|---|
| cat9_off | 0 | clean captured | 412s | tsr=0/inert + FIX-1/2/3 + chase=0 | 5/5 rc=0 |
| cat9_on_diag | 1 | EAGER + FR13_CHASE_DIAG=1 + FR13_TCF_SELFCHECK=1 + fp32 logit capture | 337s | tsr=1/armed + chase=1; iv locator=1 module; H3 layer pinned | 3/3 rc=0 |
| cat9_on | 1 | clean captured | 417s | tsr=1/armed + chase=0 | 5/5 rc=0 |
| chain5_off | 0 | clean captured | 413s | tsr=0/inert | 5/5 rc=0 |
| chain5_on | 1 | clean captured | 417s | tsr=1/armed | 5/5 rc=0 |
| cat9_off_b | 0 | clean captured | 412s | tsr=0/inert | 5/5 rc=0 |

Zero FAIL lines on any arm; fail-loud scans clean (FIX-1/2/3, FIX-A1's three
raises, KV/H3 vacuity raises); container envs asserted per arm incl.
`FR13_TREE_SAMPLE_ROW` and `FR13_TREE_REQKEY=1`.

## Gates

| gate | verdict | raw |
|---|---|---|
| G1 same-seed repeat (class 8, within-boot) | **PASS** | greedy AND t06 pairs byte-identical, 4/4 prompts, all 5 clean arms |
| G2 h1_row_mismatch == 0/all (PRIMARY; in-process) | **PASS** | **0/232** decode events (greedy+rep2 windows); `tsr_row_eq_leaf` 232/232; rowbug-class events 162, rowOK 70; prefill_stale_join 8 (known class-12, excluded) |
| G2b stock-math inversion sanity | **PASS** | `offset != prev_accepted_len` on exactly the 162 ROWBUG events (= deviation-5 of the impl report; the old stock-math check is superseded by G2 as the selfcheck) |
| G3 B_JOIN byte verdicts | **PASS** | BYTE_EQUAL 224 (162 with prev_rowbug=True), BYTE_DIFF 0, READ_ROW_NOT_WRITTEN 0, NO_PRIOR_WRITE 8 (first verify/request x 2 windows, expected) |
| G4 instrument (iv) banks REAL hashes (FAIL-1 must not recur) | **PASS** | locator needle "1 kv-cache module(s)" (`model.layers.0.self_attn.attn`, bare-tensor cu130 form); 240 drafter_kv records, 0 vacuous, **3840 hashed rows, 1000 distinct k_sha4096**; H3 240 records / 719 rows |
| G5 chain neutrality | **PASS** | chain5_on accept greedy 2.8074 (band [2.6596, 3.0078]); within-boot identity 4/4 both temps; ON-vs-OFF cross-boot forks {90,59,27,117} vs floor pair {17,15,21,61} — no sub-floor fork (GB10 BI=0 floor class); construction claim (chain leaf row == L == stock) needle-confirmed armed with zero behavior shift beyond floor |
| G6 speed sanity | **PASS** | cat9_on greedy s/fwd 0.227855/0.227866 vs OFF pool [0.227426, 0.227182, 0.225774, 0.225727]; worst-vs-OFF-max **+0.44 ms**, mean **+1.33 ms** (allowance 2 ms); warm TPS 13.9-14.0 -> **18.27** (accept-driven) |

## Fingerprint table OFF vs ON (greedy windows; transition table, class 12 observational)

| arm | ROWBUG-class root-match | rate | LCP | rowOK rate | L>=3-prev rate | L>=3 LCP | root-reject% | transitions |
|---|---|---|---|---|---|---|---|---|
| cat9_off (tsr=0) | 28/67 | 0.418 | 1.12 | 0.800 | 0.390 | 0.97 | 39.7% | 142 |
| cat9_off_b (tsr=0) | 39/69 | 0.565 | 1.52 | 0.794 | 0.542 | 1.53 | 34.0% | 137 |
| **cat9_on (tsr=1)** | **81/90** | **0.900** | **3.17** | 0.862 | **0.883** | **3.04** | **13.8%** | 119 |
| **cat9_on_diag (tsr=1)** | **74/78** | **0.949** | **3.21** | 0.806 | **0.957** | **3.26** | **12.4%** | 109 |
| chain5_off (control) | — | — | — | 0.891 | 0.918 | 3.05 | 13.5% | 129 |
| chain5_on (control) | — | — | — | 0.863 | 0.857 | 3.24 | 16.3% | 131 |

ROWBUG-class transitions now MATCH/EXCEED rowOK-class and the chain control
(the bind's predicted ~0.85-0.9); rowOK-class rates unchanged (the fix is a
no-op there by construction). ROWBUG transition share rose 47-50% -> 72-76%
(more deep accepts => more L>=3 prev events), as expected.

## Accept/event DRAWS (clean captured arms; recorded, never gates)

| arm | greedy (x2 reps identical) | t0.6 (x2 identical) | greedy s/fwd | per-depth hist greedy 0..5 |
|---|---|---|---|---|
| cat9_off | 2.0274 | 2.0059 | 0.227426 / 0.227182 | 58/13/15/17/13/30 |
| cat9_off_b | 2.2482 | 2.0833 | 0.225774 / 0.225727 | 48/11/21/18/5/38 |
| **cat9_on** | **3.1789** | **2.9615** | 0.227855 / 0.227866 | **17/6/21/18/16/45** |
| chain5_off | 2.9098 | 3.2195 | 0.224037 / 0.223742 | — |
| chain5_on | 2.8074 | 2.9308 | 0.221901 / 0.221828 | — |

cat9 OFF draws sit in the standing 2.01-2.26 band; cat9_on greedy 3.1789 is
above the chain5 band and at/above the banked native-E5 reference (3.16 bar,
not re-run) — a DRAW, not a gate; the deliverable e2e-vs-E5 comparison stays
its own campaign. t0.6 s/fwd carries wall jitter (greedy is the speed basis,
class 12).

## Served-stream evidence (greedy; floor-bracketed, class 11)

- cat9 ON-vs-OFF forks {17,15,21,70} vs the OFF-vs-OFF floor pair
  {17,15,21,61}: the fix's served-stream divergence is **itself floor-class on
  3/4 prompts** despite completely different drafts (accept 2.03 -> 3.18) —
  exactly what verify-side losslessness predicts at greedy (served tokens
  stay target-argmax; committer math untouched by FIX-A).
- cat9_on_diag vs cat9_on forks {34,11,63,61} = the known eager-vs-captured
  substrate fork class; diag numbers never quoted as baseline.
- Pre-fork counters (cat9-vs-chain5, same served prefix): OFF events-before
  cat 11/7/8/22 vs chain 10/5/9/16 with cat accepted-before deficit (24/8/16/46
  vs 24/12/16/51); ON: cat 6/12/6/16 vs chain 6/11/6/19, accepted 15/23/16/52
  vs 11/26/15/49 — the systematic cat event-count excess/accept deficit is
  GONE (converged within floor wiggle).

## Diag-boot instruments (cat9_on_diag, eager; greedy+rep2 windows)

- rowtrace: 240 records = 232 decode + 8 prefill-stale-join (offsets =
  prompt_len-1, the step-1 class-12 exclusion). **h1_row_mismatch 0/232**.
- B_JOIN: as G3. GDN h0 handoff stays byte-equal under the fix, including
  after all 162 rowbug-class commits.
- H6/TCF selfcheck rode along: **mismatch=0** at 11000 checks x 8 stages
  (committed_read_cols .. conv_new_state), zero MISMATCH lines; CV tap 232 rows.
- Instrument (iv) REPAIRED + ENGAGED: `model.layers.0.self_attn.attn`
  (kv_shape [2,1085,832,4,256]), 16-row tail window per event, real sha4096
  (1000 distinct K hashes / 3840 rows). Banked for the FIX-A2/H2 decision.
- H3 probe (minimal, deviation in-band): target
  `language_model.model.layers.11.self_attn.attn` (16 candidates, gid 3,
  kv_shape recorded). 719 depth-rows: foreign_slot by depth d2..d5 =
  167/130/84/76 foreign vs 16/17/22/4 clean (d1 203/0) — the predicted
  TOPOLOGICAL pattern (full-spine accepts foreign at every depth>=2 because
  same-depth tree rows share one slot; alt-leaf accepts clean; chains never
  foreign). Deviation recorded in every record: served-slot only, accepted-row
  pre-overwrite K/V not captured (needs in-forward tap), block table =
  drafter kv group.

## Deviations / open items (explicit)

1. H2 cat-vs-chain KV-hash JOIN not yet run: the arm list contained no chain5
   diag boot (step-1 bind suggested it "can ride along"; today's task arms
   1-6 did not include it). The cat9 side is now REALLY banked; a chain5 diag
   boot remains a cheap rider on any future diag campaign if FIX-A2 is
   pursued.
2. H3 stays the minimal/topological version (task-allowed); per-record
   `deviation` field documents the served-slot-only limitation.
3. The fingerprint/normalization rows are observational (class 12)
   conditional rates; the exact claims are the in-process integer/byte gates
   (G2/G3) and within-boot identities (G1).
4. accept 3.1789 vs native bar: cross-boot/floor caveats apply; the e2e
   lossless+superset verdict vs E5 (B=4 captured SWE-4) is NOT claimed here.
5. `FR13_TREE_SAMPLE_ROW` remains default OFF (launcher); flip is a
   user/monitor decision on this bind.

## Working-tree state (for the monitor)

Modified (uncommitted): `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (FIX-A1
+ committer NROWS publish x2 + REQKEY clear + iv locator repair + H3 +
FAIL-2 'rows' restore), `scripts/fr13_launch_forked_fa2_tree_server.sh`
(FR13_TREE_SAMPLE_ROW/FR13_CHASE_H3/FR13_CHASE_H3_LAYER/
FR13_CHASE_KV_ALLOW_EMPTY defaults + docker passthrough),
`tests/test_fr13_chase_diag_wiring.py` (16 tests). Untracked:
`tests/test_fr13_tree_sample_row_wiring.py` (15 tests),
`output/fr13_chase_fixA/` (campaign artifacts, gitignored), this bind.
