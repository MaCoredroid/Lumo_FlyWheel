# Common-O0 repaired A+B: full independent raw closure

**PASS: 336/336 observations authenticate, and all 84 cases satisfy the frozen target-only decision prerequisite across A/B × r0/r1.** Every case has the exact fixed common O0, identical resulting O1, identical full-vocabulary logits, and the same smallest-ID greedy decision across all four observations. No missing case, tolerance change, tie-policy change, substitution or denominator reduction was used.

This closes the tested **native target-only common-O0 cycle-0 calibration prerequisite**. It does not qualify candidate verification, MTP, joint target/MTP state, continuous cycles, held-out inputs, lifecycle behavior, serving throughput or agent workloads. The original natural-prefill A/B adverse result remains 0/84 eligible under its original criterion, including 14 differing greedy decisions; the failed common-O0 v1 attempt and both early review seals remain preserved. A controlled shared-state pass does not erase those outcomes or establish their cause.

## Complete result

Cohort: `q1-native-common-o0-repair2-20260929T125100Z`, aligned_nonpacked processes A and B, two repetitions each.

| Quantity | A | B | A/B × R2 |
| --- | ---: | ---: | ---: |
| Expected / authenticated observations | 168 / 168 | 168 / 168 | 336 / 336 |
| Complete case groups | 84 | 84 | 84 |
| Common O0 equality | 84 | 84 | 84 |
| O1 equality | 84 | 84 | 84 |
| Full-logit byte equality | 84 | 84 | 84 |
| Stable exact greedy decision | 84 | 84 | 84 |
| Exact-tie observations | 2 | 2 | 4 |
| Per-run unique raw objects checked | 107,396 | 107,400 | 214,796 summed |
| Raw object bytes checked | 27,127,390,208 | 27,133,804,544 | 54,261,194,752 summed |
| Terminal members size/hash verified | 374 | 374 | 748 |

All four ties belong to `calibration-long_available__c0__n05`: winner 279, tie count 2, margin 0. Its four full logit exports have the same SHA `8326cd231955c54eed1fb3de103ab50cd6514cb79c61bc0ee64c23da03a64ece`. Full-logit equality refers to the complete 248,320-value float32 raw export of the bf16 head output. Every referenced raw object passed hash, length, geometry/dtype and finite-value checks. Object totals sum the distinct objects within each run, rather than claiming global deduplication across A and B.

The exact unmodified `q1_native_common_o0_corpus_v2.reduce` authenticated both full runs. No AST predicates were omitted in this paired reduction. The strict target-only `q1_native_common_o0_audit_v2.repeat_summary` independently agrees: denominator 84, complete cases 84, common-source O0 matches 84, decision-prerequisite cases 84. The reducer's legacy field `primary_baseline_qualified=true` refers to this scoped target-only corpus; it must not be read as full Q1 qualification. The independent result explicitly leaves full Q1, MTP, candidate, held-out and workload flags false.

A's rows and run binding reproduce its accepted full audit exactly. Its sealed authentication of all 84 original A/r0 source records and O0 objects is reused unchanged. Every new observation's imported O0 is independently checked against the same immutable all-84 source mapping. `ALL-CASE-DECISIONS.json` preserves every process/repeat winner, tie count, margin, raw-logit SHA and state digest; no adverse records are filtered.

## Runtime provenance and termination

All 27 frozen source payloads match. Both corpus/job reconstructions, actual image/config/command/environment/mount/resource bindings, native runner patch reproduction, FA2 installation and boot route, driver parity/seals, gate and readiness bindings, and terminal members pass the frozen checks. This uses the v2 metadata repair only; no future joint target/MTP source enters either run or this audit.

Image: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. A CID `ece7c243f6ee7855dd9fe6eeb83add7d2bebf7aa46b1719ecdd6641d6f828809`; B CID `fbb789e46642e46391e099869ab89b419a9f4f8093b169158eefd6029b065f42`. They are distinct fresh processes. Each created/stopped/removed CID matches exactly; final retained inspect shows not running, exit 0, no OOM, no restart, and empty cleanup error. Both terminal statuses are `COMPLETED_driver_rc=0_cleanup_rc=0`.

B boot: 14:08:31–14:13:42 UTC (311 seconds). Driver: 14:13:42–15:01:59 (2,897 seconds, 48m17s). Cleanup stopped the owned container at 15:02:01. A boot was 312 seconds and driver collection 2,783 seconds (46m23s). These are qualification operational intervals, not performance measurements.

The exact gate-bound backup and completed readiness receipts authenticate for both processes. B's backup receipt SHA `f85e437dc179c23f66f3fcde9bd201eb088a8d1ace6a49a8fdf21a9ef7af587a` binds commit `25bb8dda7c161cb89bef3728cceccf9b43b36aa0`; the recorded remote/archive verifications pass. The reviewer authenticated retained evidence and did not repeat a backup or network/LFS operation.

## Independent evidence

The CPU audit ran 15:07:01.822609–15:08:49.295130 UTC: 107.473 seconds with OMP/OpenBLAS/MKL ceilings 2. It ran only after both terminal receipts existed. No GPU/model/engine/Docker/cache operation, source edit, or gate mutation was performed. The laptop received only summaries, selected early JSON, source snapshots and terminal metadata; no full raw object store was copied locally.

Directory on both hosts: `p0/monitor/review-response-20260927/native-common-o0-repair2-b-independent-20260929/paired-attempt1/`. Key files are `RESULT.json`, `FROZEN-REDUCTION.json`, `TARGET-ONLY-REPEAT-SUMMARY.json`, `ALL-CASE-DECISIONS.json`, `RUN-METADATA.json` and `MANIFEST.json`. The wrapper and exact official CLI arguments are in the parent review directory. All local copies were checked against the remote audit manifest.

SHA-256 bindings:

- Frozen corpus reducer: `bc4e04b0b6cabcc0a4f10ab63e9fd3859ce2721a22dac0a9505bf441362c1c37`.
- Independent wrapper: `8e04e585bd55ff2a11df9731c47a41aac05baf224307f936f8c2db8040792307`.
- B terminal: `a2cb7a36bb497534a0c9fde8fa2844eb163e858b53259c2691a6a4e30a3de3a1`.
- Paired result: `0c787fd1e92abd0b39ca05978638956c03562aa527efbe9feaa80a01b7f65cc3`.
- Full frozen reduction: `653d027d8a1f03643518d65c00ad69797c0c2f135ddd88262c30a86ff925033e`.
- Audit payload manifest: `52a090538ff795af6ccded7663c49d9d59beefe4af562b4ef9184a570b3b6cf3`.
