# Common-O0 B: early raw check and paired-reduction preparation

**The four early B observations pass; the complete B run and 84-case A/B result remain pending.** I checked the same preselected short-prefix root-only and n02 cases as A, both r0/r1 repetitions, using the unchanged frozen v2 raw auditor. There was no selection based on B outcomes. No GPU, model, HTTP, Docker, cache, backup or implementation operation was performed.

Run: `q1-native-common-o0-repair2-20260929T125100Z-aligned_nonpacked-B`. CPU audit time: 14:18:04.046692–14:18:06.566891 UTC, 2.520 seconds, OMP/OpenBLAS/MKL ceilings 2. At both ends, 33 case files existed and no terminal receipt was present. Only four B observations were independently audited.

| Early case | B r0/r1 winners | Ties | Margins | A/B × R2 comparison for this case |
| --- | --- | --- | --- | --- |
| short/root-only | 13, 13 | 1, 1 | 0.25, 0.25 | common O0, O1, full logits and greedy identical |
| short/n02 | 198, 198 | 1, 1 | 0.5625, 0.5625 | common O0, O1, full logits and greedy identical |

The A rows were reused only after verifying their binding to A's immutable full-review seal `47beb72a…`. The early comparison covers two of 84 cases and makes no corpus-level stability claim. No new defect was found in this bounded sample.

The check authenticated 14,134 current-run objects (2,546,962,432 bytes), including dtype-specific finiteness and full raw logits; separately authenticated both original source record seals and their shared O0 objects (6,848 objects, 1,037,828,096 bytes); verified the fixed O0 manifest, source/worker/owner/control/chain and import bindings; and checked all 27 frozen payloads. The complete existing v2 auditor, not a shortened numerical check, was used for each selected observation.

B's gate hash is `edb36be728ae60e18a6da2acc9158ebf33a307eec41dee77aa79412dba60bb79`. Its job hash is `d899c29849b2ae836a6b01620c44622a093060d14d76491cce638250a811e41f`. Gate, launch, job reconstruction, rendered config, stock-runner patch reproduction, image, FA2 installation/worker dispatch, boot PID, and readiness evidence agree. The retained created CID is `fbb789e46642e46391e099869ab89b419a9f4f8093b169158eefd6029b065f42`; image remains `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. Recorded engine start was 14:08:31 and healthy/driver start 14:13:42 UTC: 311 seconds of boot, not a performance measurement. Final actual-container configuration and cleanup verification await terminal evidence.

The exact gate-bound backup receipt `f85e437dc179c23f66f3fcde9bd201eb088a8d1ace6a49a8fdf21a9ef7af587a` binds commit `25bb8dda7c161cb89bef3728cceccf9b43b36aa0`. It records a matching remote branch, successful archive decrypt/member checks including repaired A, and HTTP 200 checks for every listed LFS object. This review authenticates that retained receipt; it does not claim to have repeated network queries, downloads, decryption or backup work.

The paired CPU closure is prepared as `audit_paired.py` in `p0/monitor/review-response-20260927/native-common-o0-repair2-b-independent-20260929/`. It requires both terminal receipts, calls `q1_native_common_o0_corpus_v2.reduce` directly without AST exclusions or criterion changes, reproduces A's accepted rows, and joins all 84 × A/B × R2 observations. It additionally preserves exact ownership/cleanup, backup/readiness, per-process counts, every decision/tie/margin and the strict target-only summary. All 84 cases remain in the denominator even if cross-process decisions differ. A result can authenticate completely while still failing the target decision prerequisite; full Q1/MTP/candidate/workload flags stay false either way.

`PAIRED-PREPARATION.json` records the exact frozen official CLI argument array and the independent wrapper command. The full paired reduction has **not** run. The reviewer will wait for the parent's terminal notice without polling. Future joint target/MTP sources are not used by this run or either audit. Original natural-prefill failures, the failed common-O0 v1 attempt, and A's full/early review seals remain preserved.

Hashes:

- Frozen corpus reducer: `bc4e04b0b6cabcc0a4f10ab63e9fd3859ce2721a22dac0a9505bf441362c1c37`.
- Early B audit script: `07cec95b29029385b392df6444bc9060e572039544864e7bed82f3dd35ad32e1`.
- Early B result: `aea0df63ffb53ee6fd16648bc6ac092af381f48c6b0af52cedcb94dbd319370b`.
- Early payload manifest: `aeef8c8d055e59ae294cff21da0fc05897d8c4dc68cc77a833d01a63a19bddfe`.
- Prepared paired wrapper: `8e04e585bd55ff2a11df9731c47a41aac05baf224307f936f8c2db8040792307`.
