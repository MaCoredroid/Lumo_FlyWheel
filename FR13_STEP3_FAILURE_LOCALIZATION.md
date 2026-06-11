# FR13 Step 3 Failure Localization

Date: 2026-06-11 UTC

Base commit: `f8e35e7a974bb620d719a077424e8b0547efd018`

Scope: post-handoff B=4 losslessness/corruption failure after the replay-route
serving crash fix `37a98fbd`.

## Verdict

No small confident code fix is supported by the Step 3 evidence. The current
failure is a real B=4 tree-verifier quality deficit, not a replay-route serving
crash, not a prompt-pairing artifact, and not just the known p0@35 near-tie.

The next discriminator should be a focused verify-drift/acceptance trace, not
another broad B=4 gate:

1. Re-run a diagnostic pair with native `per_req_spec_trace.jsonl` enabled
   (`FR10_METRICS=1` or an equivalent trace-only switch), keeping the Step 3
   prompts/seeds/config fixed.
2. Pair tree `tree_path_lcp.jsonl` spine-prefix acceptance against native
   accepted lengths to classify the accept deficit by depth.
3. If the paired trace points at verify drift, run the existing teacher-forced
   fixed-prefix/logit path (`scripts/measure_spec_teacher_forced.py` plus
   `FR13_FINAL_LOGIT_CAPTURE`) around the earliest real failing positions to
   compare the tree spine rows `[0,1,3,5,7]` to native MTP-5 rows on identical
   prefixes.

## Evidence Read

Primary gate:

- Artifact: `output/fr13_step3_b4_gate/fr13_corruption_gate.json`
- Comparator: current E5 native K=5 arms.
- Prompt identity: `tree_vs_native=true`.
- Tree: `TREE_ATTN/tree_mtp`, seed `1313`, B=4, active width 9, 16 records,
  accept/event `2.132045088566828`.
- Native: `FLASH_ATTN/naive_mtp`, seed `1313`, K=5, 16 records, accept/event
  `2.7830882352941178`.
- Native noise: `FLASH_ATTN/naive_mtp`, seed `2313`, K=5, 16 records.
- Native self bag-TV: `0.19677734375`.
- Tree-vs-native bag-TV: `0.2438904313016529`, above both the current reducer
  budget `0.19677734375` and the documented `0.113` floor.
- Real-loss rate: `0.28378378378378377` > `0.05`.
- Depth-collapse detector fired: prompt `0`, run length `6`, ending position
  `59`.
- Accept drop: tree-native delta `-0.6510431467272899`.

Replay-off diagnostic, re-reduced against current E5 K=5 native arms:

- Command:
  `python3 scripts/fr13_corruption_gate.py --tree-run output/fr13_step3_b4_gate/tree_diag_replay_off --native-run output/fr13_step3_b4_gate/native --native-noise-run output/fr13_step3_b4_gate/native_noise --out output/fr13_step3_b4_gate/fr13_corruption_gate.diag_replay_off_vs_e5.json`
- Exit code: `2` (valid FAIL).
- Tree replay-off accept/event: `2.1498316498316496`.
- Native K=5 accept/event: `2.7830882352941178`.
- Accept delta: `-0.6332565854624681`.
- Bag-TV: `0.23340632148143167` > `0.19677734375`.
- Real-loss rate: `0.5075757575757576`.

The pre-existing `fr13_corruption_gate.diag_replay_off.json` compared
`tree_diag_replay_off` against the archived K=9 native arms, so it is not the
E5 K=5 verdict. It remains useful only as historical context.

## What This Localizes

The p0@35 first loss in the primary gate is the known cross-boot near-tie
confound:

- Current first real loss: prompt `0`, position `35`, tree `8445`, native
  `44675`.
- Banked docs already reclassified this exact site as a cross-boot near-tie
  rather than a stable replay/legacy defect (`FR13_REPLAY_GPU_GATES_BIND.md`).

The Step 3 failure is broader than that site:

- Prompt 0 has `37/90` eligible outside-self-noise losses.
- Prompt 2 has `5/26` eligible outside-self-noise losses.
- Bag-TV exceeds the native-self budget.
- Accept/event is lower than native by about `0.65`, independently failing the
  summary accept gate.

The tree branch path is active and locally supersets its own spine:

- `tree/logs/tree_path_lcp.jsonl` has `647` sampled accept rows.
- Accepted length histogram: `0:241`, `1:52`, `2:85`, `3:90`, `4:46`, `5:133`.
- The true native spine path for the 9-node caterpillar is `[0,1,3,5,7]`.
- Spine-prefix mean from the tree trace is `1.8346213292117466`.
- Accepted-length mean from the same trace is `2.072642967542504`.
- Mean branch bonus is `0.23802163833075735`, with `154` positive-bonus events
  and no events below the tree's own spine prefix.

So the internal tree branch selection is adding opportunity, but it is adding
opportunity on top of a degraded B=4 tree trajectory/spine rather than matching
the native MTP-5 reference.

## Ruling Out Known Wrong Roots

- Not a serving crash: the replay-on tree arm completed FULL capture and
  produced 16 records.
- Not prompt pairing: reducer prompt identity is byte-identical.
- Not K=9 native comparator contamination in the primary verdict: native and
  native-noise active width is K=5, with draft ratios exactly `2720/544 = 5`
  and `2620/524 = 5`.
- Not fixed by replay-off: replay-off vs current E5 K=5 still fails with a
  similar accept drop and bag-TV failure.
- Not branch-disabled: branch bonus is positive in the tree trace.

## Next Discriminator

The missing discriminator is per-event native acceptance/verify evidence. Step
3 launched with `FR10_METRICS=0`, and the scheduler spec trace only writes
`per_req_spec_trace.jsonl` when `FR10_METRICS=1`, so
`per_event_superset=null` in the reducer.

Run the smallest live follow-up as a diagnostic, not a verdict gate:

1. Tree arm: same Step 3 tree config, keep `tree_path_lcp.jsonl`.
2. Native arm: same Step 3 E5 K=5 config, enable `per_req_spec_trace.jsonl`.
3. Reduce with `src/lumo_flywheel_serving/fr10_superset_gate.py` or
   `scripts/fr10_superset_gate_report.py` to report native accepted depth,
   tree accepted depth, and tree spine-prefix depth.

Interpretation:

- If tree spine-prefix depth is below native while tree spine draft tokens match
  native at the first degraded depth, localize to verifier/position/commit
  bookkeeping.
- If tree spine draft tokens differ at the first degraded depth, localize to
  draft-side recurrent-state or trajectory contamination.
- If per-event acceptance is inconclusive because sampled paths diverge too
  early, switch to teacher-forced fixed prefixes around p0@35 and p2@21-26 and
  compare final-logit distributions on identical prefixes.

Do not launch the next broad FR-13 gate until that discriminator binds the
deficit.
