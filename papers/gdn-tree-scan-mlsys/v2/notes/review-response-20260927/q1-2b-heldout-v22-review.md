# Q1.2b held-out extension v2.2: bounded source review

2026-09-28 UTC. **PASS for the reviewed changed-scope executable logic, conditional on the parent's final freeze comparison and stated calibration-power gate bindings. No remaining material executable blocker found. This is preparation review, not launch authorization or held-out qualification.** No GPU, Docker, model, remote command, real held-out fixture tensor, or candidate implementation was executed by this reviewer. Frozen sources and run evidence were not changed.

## Reviewed identities

Sources are relative to `experiments/review-response-20260927/`:

| File | SHA-256 |
|---|---|
| `tools/q1_component_runner_v2_2.py` | `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29` |
| `tools/q1_2b_reduce_v2_2.py` | `6475182d9e0996dd4e7c228340c57299665cd1ea641ed0b61bf291d631af9d73` |
| `tools/run_q1_2b_component_v2_2.sh` | `5cc0fdd7fc0895560de77e8046c73c7eac3f010bba8eee137b3573d3eaf050b5` |
| `tools/tests/test_q1_2b_v2_2_heldout.py` | `06ef9bd4825825a41de3811b024b680bce998a317f17c7208b70a122d58ad0bb` |
| completed `tools/test_log.q1_2b.v2_2_heldout.attempt1.txt` | `435eda6a37f366890d9f5e75086722422c668792423d65e37cc10e27fdcad5a0` |

The accepted v2.1 comparison identities remain runner `b7241e8b925a9b5d9e921673d84f84b701f9b1404ed71d727bc5ade37d15cb65`, reducer `1d58c83d4cf96d755d12822ed23f4557e5d037f6f9aeadbb8a135dfe5d59449c`, launcher `84a377973c3cf68504b964fae7082e8db4d4daba6335bb9878f02bc3cb2456e2`.

An exact independent source snapshot is retained in `notes/review-response-20260927/heldout-v22-review-snapshot/`, with `SNAPSHOT.json` SHA `5762ba964e1eb05789e99fc07ea0b7a16b7589538148d8432807d662ff30b9d1`. Its initial test-log copy intentionally ends before pytest output; `completed-test-log.txt` separately preserves the later complete log above. No initial snapshot was overwritten.

## Changed-scope checks

1. **Frozen evaluation coverage.** The runner adds only `evaluation` to the prior calibration block and checks manifest IDs against the policy's `evaluation_held_out` IDs and the gate's exact named IDs. Partial/duplicate/missing IDs, a nonzero limit, nonzero held-out negatives, wrong block and wrong runner bytes refuse (`runner:477–520,830–839`). Actual metadata yields exactly `evaluation/fx0_ordinary-random` and `evaluation/fx1_mixed-stress`: 5,760 unique cases, comprising 3,072 outputs and 2,688 states, or 276,480 head observations per process. No output or state denominator was reduced.

2. **Full A/B support and source identity.** Launcher fixes repeats 2, evaluation block, negatives 0, separate held-out output root and parent gate, and still performs sequential fresh A/B execution with failure preservation. Reducer binds its own bytes and the exact v2.2 runner/launcher constants; source/helper maps must match gate authority (`reducer:201–234`). Gate IDs, manifest IDs and policy IDs must agree (`275–283`). Both processes still require exact reference/eligibility/metric case sets, complete fixture products, raw record/tensor hashes, reference-first chronology, native/candidate within- and cross-process determinism, publication path reconciliation and the original paired numerical criterion (`350–420` and unchanged later checks). Neither omitted process nor missing records become a held-out PASS.

3. **Negatives remain calibration-only.** Held-out scope requires zero negatives; any held-out negative inventory record, fixture negative product or negative-only structural flags refuses (`reducer:391–398,439–450`). The only removed requirements are the calibration-specific negative product/power checks. N5 and negative computations themselves were not modified. Negative power remains enforced for calibration (`594–605`).

4. **Numerical/replay behavior unchanged.** AST comparison shows all 33 pre-existing runner functions/classes except `main` unchanged; the only new function is `block_authorization`. This includes backend, public boot lifecycle, reference-first fixture loop, native reference generation, production publication, negative interventions, storage and metric functions. All 11 reducer functions except `reduce_run` are unchanged, including `_recompute_metrics` and `verify_record`. No evaluator, threshold, oracle, kernel, topology or fixture generation change is in this extension. The pinned-image CPU-only `--recompute all` path and exact C2/metric-array checks remain intact.

5. **Recorded authorization label.** The inherited `authorization.scope` string still says `calibration` for any non-init numerical scope. This is descriptive inherited metadata, not the block selector: actual args, `block_authorization`, fixture selection, policy bindings, launch scope and reducer all explicitly require `evaluation`. Do not use that inherited label alone to describe a held-out result; no executable repair is needed for this bounded change.

## Calibration-power provenance: resolved by explicit parent authority

The first source inspection found that `calibration_power_run_id` is checked only as a nonempty string (`runner:515–516`, `launcher:74–75`, `reducer:215–217`); the reducer does not itself reopen the prior calibration run. A pure CPU function probe confirms a different nonempty ID also passes that local scope helper. That is **not a bypass of the trusted parent gate**: the launcher/reducer require the same hash-bound approved gate, and the parent—not an untrusted runtime record—selects its scientific authority.

The parent has now created and this reviewer verified the three copied evidence files in `p0/monitor/review-response-20260927/heldout-calibration-power/`:

- `MANIFEST.json`: `a9c1cdc3cc561b90b91ef45129f0e6c986ac39d4a0702132455b9f71936d1505`.
- Calibration summary: `68408fc1f530ca7f8de37eb4689f5becc541adfea0b2a21905838ce60e1141f6`.
- Terminal receipt: `4deaafb73996aca5260ce0b2d1aeccd45edf6dc7daa42c3799eab47d68262583`.
- Parent acceptance: `9d56a48cd15bc981536a2059451af0d638fc39da899b892636ce7e2d5e3963cc`.

All three size/hash values match. The copied summary and receipt establish the exact accepted run `q12b-calibration-retry-20260928T010045Z`, completed A/B/reducer exit 0, 5,760 PASS cases each, no findings, and N1/N3/N4 FAIL counts 96/4/96 in each process. N5 causal equality was independently established in the calibration review. The parent's manifest lists unchanged policy, fixture/observation family, kernel, native module, numerical helpers and topology.

**Final gate condition retained:** the parent must compare these authority hashes against the final held-out freeze and bind the exact calibration run ID, manifest and all three receipt hashes into the approved held-out gate. Under that stated trust model, this safely closes the provenance concern without modifying the scientific code or rerunning negatives. Mutating a trusted approved gate is not an untrusted runner-record test. This review does not claim that the package independently makes the parent's scientific acceptance decision.

## Executed checks and limits

The independent `check_scope.py` uses only Python's standard library. It extracts the actual source function AST, reads only frozen manifest/policy/observation JSON metadata, executes 15 targeted authorization controls, compares v2.1/v2.2 function ASTs, verifies the complete 5,760-case product, and checks shell syntax. It completed exit 0. Controls include incomplete/duplicate IDs, wrong policy IDs, wrong runner bytes, unsupported/wrong blocks, nonzero limits/negatives and absent calibration authority; the nonempty-authority control explicitly records the parent-gate boundary above.

```sh
python3 -B notes/review-response-20260927/heldout-v22-review-snapshot/check_scope.py > notes/review-response-20260927/heldout-v22-review-snapshot/independent-checks.json
```

- Script SHA: `fe2a76ea81ebb516b96d59e98ce91ce94ccbeaaa6e3a4cbb1aee6baac485c59b`.
- Output SHA: `79bc12afae062ace0698b3e4b72592f8444ea0bb96187975557359b6c353a469`.

Local Python has neither Torch nor pytest, so I did not independently rerun the author's full synthetic CPU pipeline. The completed supplied log records **49 passed in 136.63 seconds**, finishing `2026-09-28T01:27:03Z`. I independently verified all **19 start/end/current source hashes** and the three calibration evidence member hashes; compact results are in `heldout-v22-review-snapshot/final-receipt-checks.json`. The synthetic tests cover v2.2 complete A/B reduction, source/scope/negative refusals and preserved calibration behavior; they are infrastructure checks, not held-out candidate qualification. Actual final freeze acceptance and launch authority remain with the parent.

Final freeze cross-check: `experiments/review-response-20260927/FREEZE-Q1_2B-HELDOUT-v1.json`, frozen `2026-09-28T01:29:33.313667Z`, SHA `84662b3600c739ae289d080bb2fa25f951b9deb55a2017fa7e258a6670d3dbee`, contains the same reviewed runner, launcher, reducer and completed test-log hashes. I verified those four exact members against current local bytes. The parent's full 31-member local/remote freeze check and gate issuance remain separate; no executable delta appeared between this review and that submitted freeze.
