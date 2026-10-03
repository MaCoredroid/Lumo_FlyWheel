# Native aligned B and A/B: complete raw review

**B passes the complete raw audit (168/168, zero integrity/provenance failures), but aligned A/B repeatability fails the frozen criterion.** Each process is internally stable across r0/r1. Across fresh processes,14/84 cases change greedy winner, and all84 have different O0, O1 and full-logit hashes. The unchanged reducer's aligned `baseline_qualified` count is therefore **0/84**, because identical O0 is required in addition to a stable greedy decision. No case or tie was removed, and no criterion was relaxed.

This covers aligned A/B × r0/r1 (336 observations). The four-run aggregate including the separate packed controls has not been executed; packed cannot replace the aligned primary. The zero aligned count is a result on its complete fixed denominator, not an artifact of missing packed observations.

## B raw and operational evidence

Run: `q1-native-corpus-calibration-20260929T045821Z-aligned_nonpacked-B`. Its terminal receipt is `COMPLETED_driver_rc=0_cleanup_rc=0` at11:40:30.427030UTC, with driver168/168 authenticated observations and owned engine stopped/removed. Retained inspection records exit0, restart count0, not OOM-killed. The immutable image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`; container ID is `0f62d0f41cb8011a6f1710cc53490e47d14b487854a95d3de9c651208cbb368e`.

The prepared B adapter was verified locally and remotely against preparation manifest `e3486bacef2e2ebb48c670b17fcad251619e9e827adcdb4afa02f81d070ca049`. It changes only exact A/B run/process/output-schema identifiers from the accepted A audit. It uses the unchanged per-run loop of frozen reducer `d4502b7f47ab1ac6ed8cd295aa5565726fad486c4a1ff1bda6a13fe24e4ea770`, omitting only the four-directory precondition and final aggregate. No authoritative implementation was edited.

The remote CPU audit ran11:41:49.005304–11:42:20.852776UTC (31.847s), with OMP/OpenBLAS/MKL ceilings2. It authenticated all **58,578 unique raw objects /19,923,091,456 bytes**, exactly matching the terminal store, and all **374 terminal metadata members**. Every expected case, request/control/driver binding, consumed chain/position, pending-token boundary, O0/O1 state geometry/group/logical digest/storage continuity and full248,320-entry logit vector passed the original checks. All raw states/logits were finite; smallest-ID argmax, ties, top values and margins matched their records. No extra/missing observation was accepted.

Boot/image/config/source/patch/FA2 and actual Docker environment, command, mounts/resource policy passed the unchanged frozen validation. The retained SUCCESS readiness receipt binds the same run/gate/source, exit0, finalization and all boot conditions; its SHA is `c4e5f2b500721c7a9d2b2603b4ff67b2799046fbcff48d977fdb6f55c503cb38`. This reviewer performed no recovery, model, GPU or container operation.

B terminal SHA: `6f3718ac0cf20d66fdfaff541a78de16195020b95bd6e1866d92f286a8cc1153`. Job SHA: `4b8507e027e811090d01eb498bd736545124c814eb7c676a2cb6728e05a17bdb`. Corpus SHA remains `f0320a57bdec8fb9aaf878313972ea391f936fe48a8e4ad46ddacb3375e40e44`. Independent B audit manifest SHA: `cb7f477dcf84add6077690c930d11471cc7155ce3951907cf9c733b099487e10`.

## Same-case comparisons

The A/B comparison authenticates both independent audit manifests/results/rows against their completed terminal receipts, then rechecks all336 case members, case seals and row-to-O0/O1/logit/decision bindings. It calls the unchanged frozen `repeat_summary` on these authenticated rows and retains every one of the84 cases.

| Property | Within A r0/r1 | Within B r0/r1 | Across A/B × r0/r1 |
| --- | --- | --- | --- |
| Complete case groups | 84/84 | 84/84 | 84/84 |
| Stable greedy decision | 84/84 | 84/84 | 70/84 |
| Identical O0 logical state | 84/84 | 84/84 | 0/84 |
| Identical O1 logical state | 84/84 | 84/84 | 0/84 |
| Identical full logit bytes | 84/84 | 84/84 | 0/84 |
| Exact-tie observations | 2/168 | 0/168 | 2/336 |

The driver's global `repeat_*_identical=false` fields compare all different cases together and remain unsuitable for this question. The cross-process differences above are independent, case-matched findings, not that summary artifact. Both A tie observations are long/n05; B has an unambiguous different winner for that case.

## Source, configuration and job comparison

No substantive recorded source/configuration difference was found. All original differences and normalization rules are retained in `ALIGNED-AB-DIAGNOSTICS.json`; none were silently dropped.

- Jobs differ only at `built_utc`, `run_id`, `process`, and the canonical digest derived from those identities. All cases, prefixes, forced chains, repeat counts, expected packed policy, model geometry, FA2 bindings and archival controls match.
- Generated engine configurations differ only in generation time and run-specific names, paths and job digest bindings. The actual Docker Config differs at hostname and the run/container/job-hash environment fields. Actual HostConfig differs only in the owned log bind and CID-file path. After explicit identity normalization, full generated configs, actual Config/Env/HostConfig and launch argv match.
- All frozen source hashes, declared geometry, boot cache description, attention implementation and installed FA2 identity match. Runner-patch and FA2-install receipts differ only in their `utc` values (A09:55:44, B11:13:33).

The normalizer substitutes only the explicit run ID, process field, actual container ID/hostname, verified job file/canonical digests, and generation timestamp fields. Environment ordering is compared as a key/value map and bind ordering as a sorted list. Raw differences are retained alongside normalized comparisons. This does not prove equality of every unobserved runtime tensor or diagnose why the processes diverged.

## Fourteen changed decisions and raw logit differences

Each table row represents both repeats: within each process their raw logit hashes and metrics are identical. Token IDs are reported without decoding. Maxabs and RMS use all248,320 raw logit entries, with authenticated little-endian FP32 storage promoted to FP64 arithmetic: `delta=A-B`, `max(abs(delta))`, and `sqrt(mean(delta²))`. Both repeats' vectors were reread, rehashed and finite-checked for this calculation. Full-precision metrics and both raw hashes remain in JSON/CSV.

| Case ID | Winner A→B | Ties A/B | Margin A/B | Maxabs | RMS |
| --- | --- | --- | --- | --- | --- |
| `calibration-short_available__c0__n02` | 198→13 | 1/1 | 0.5625/0.125 | 3.875 | 0.888370 |
| `calibration-short_available__c0__n04` | 198→279 | 1/1 | 0.4375/0.125 | 2.875 | 0.674219 |
| `calibration-short_available__c0__n06` | 25→314 | 1/1 | 0.375/0.375 | 2.78125 | 0.540814 |
| `calibration-short_available__c0__n09` | 198→13 | 1/1 | 0.59375/0.0625 | 2.26562 | 0.410354 |
| `calibration-short_available__c0__n10` | 198→13 | 1/1 | 0.5625/0.125 | 3.26953 | 0.465643 |
| `calibration-short_available__c0__n11` | 198→11 | 1/1 | 0.5625/0.1875 | 2.8125 | 0.505039 |
| `calibration-short_available__c0__n13` | 198→310 | 1/1 | 0.3125/0.0625 | 2.65625 | 0.518059 |
| `calibration-short_available__c0__n16` | 3274→198 | 1/1 | 1.25/0.4375 | 3.05859 | 0.568538 |
| `calibration-medium_available__c0__n24` | 279→198 | 1/1 | 0.1875/0.15625 | 1.78125 | 0.325942 |
| `calibration-long_available__c0__n05` | 279→353 | 2/1 | 0/0.0625 | 2.90625 | 0.420156 |
| `calibration-long_available__c0__n09` | 279→198 | 1/1 | 0.125/0.53125 | 2.01562 | 0.318098 |
| `calibration-long_available__c0__n12` | 279→198 | 1/1 | 0.1875/1.1875 | 1.875 | 0.373933 |
| `calibration-long_available__c0__n15` | 11→279 | 1/1 | 0.21875/0.125 | 1.84375 | 0.346498 |
| `calibration-long_available__c0__n19` | 279→198 | 1/1 | 0.25/1.21875 | 1.81641 | 0.290181 |

The14 changed decisions comprise eight short, one medium and five long cases. Maxabs ranges1.78125–3.875; RMS ranges0.290181–0.888370. These observed differences are not explained by the single exact-tie case. Their cause remains unestablished; no new experiment, retry, altered tolerance or reference substitution is authorized by this review.

## Evidence and disposition

All evidence is under `p0/monitor/review-response-20260927/native-aligned-b-independent-20260929/`, including the immutable preparation, unchanged frozen reducer, executable one-run/metadata/comparison/diagnostic scripts, B rows, complete84-case A/B comparison, original configuration diffs and14-case CSV with full logit errors. The original A review and outputs remain unchanged.

A/B comparison SHA: `01b9fe92697ed4d5d38b0b957d06f31a745ff4d2169fe14eabdb04d4d9908ac5`. Diagnostic SHA: `17bcc10f63570b17a720b70ce7701f64e3a408ba7b3bf2ade5dc31d06f3cb162`. A separate review manifest binds the completed evidence and this note without rewriting earlier audit/preparation manifests; local and remote copies are checked against it.

B is a valid completed characterization run. The aligned cross-process common-O0/greedy prerequisite is not satisfied under the frozen rule. This result does not by itself identify a candidate error, establish an alternate acceptable reference, or qualify MTP, lifecycle, held-out, continuous-cycle, performance or agent-workload claims. Parent retains the scientific gate and the decision about any still-authorized packed-control continuation.
