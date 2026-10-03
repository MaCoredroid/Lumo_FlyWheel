# Native common-O0 partial A/B comparison: first ten cases

**No disqualifying categorical mismatch was found in these ten cases.** Across A/r0, A/r1, B/r0 and B/r1, all ten have the same imported joint O0, matched declared phase inputs, identical target O2 smallest-ID winner, and identical actual MTP spine plus ordered top3 at each of the three declared phases. This is a **40-observation partial check**, not completed 336-observation admission. The remaining **74 cases are unchecked here**, not assigned a pass.

Selected cases are exactly the previously audited B batch1: `calibration-short_available__c0__root-only` and `n01` through `n09`, each with two repeats. The independently accepted complete-A review supplies only the matching 20 A observations. There is no outcome-based selection, additional model execution, threshold change or new population.

## Exact observed categories

Every value in each row is identical across all four observations. MTP cells show `actual spine / actual ordered top3`; target O2 is its smallest-ID argmax. The MTP spine is not asserted to equal the target model's next-token winner.

| Case | Target O2 | Before-Z first | Before-Z follow | After-Z first |
|---|---:|---|---|---|
| root-only | 198 | 314 / [314, 364, 279] | 279 / [279, 411, 593] | 248069 / [248069, 12, 9764] |
| n01 | 198 | 198 / [198, 314, 25] | 248069 / [248069, 16, 12] | 248069 / [248069, 510, 16] |
| n02 | 198 | 198 / [198, 314, 13] | 248069 / [248069, 510, 248046] | 248069 / [248069, 510, 248058] |
| n03 | 310 | 310 / [310, 321, 314] | 279 / [279, 1423, 593] | 279 / [279, 264, 593] |
| n04 | 198 | 198 / [198, 13, 314] | 248069 / [248069, 510, 248058] | 248069 / [248069, 510, 248058] |
| n05 | 198 | 198 / [198, 369, 271] | 248069 / [248069, 9764, 760] | 248069 / [248069, 510, 12] |
| n06 | 314 | 198 / [198, 13, 271] | 248069 / [248069, 760, 12] | 279 / [279, 411, 264] |
| n07 | 198 | 198 / [198, 13, 11] | 248069 / [248069, 510, 248058] | 248069 / [248069, 510, 248058] |
| n08 | 310 | 310 / [310, 198, 13] | 279 / [279, 1423, 264] | 279 / [279, 1423, 264] |
| n09 | 198 | 198 / [198, 13, 11] | 248069 / [248069, 248058, 510] | 248069 / [248069, 510, 248058] |

## Vector diagnostics and integrity

Complete raw-logit hashes are identical across all four observations in **10/10 cases for each of the four phases**, separately from categorical stability. The 40 unique full-vocabulary vectors already mirrored with B batch1 cover every selected A vector hash. This review reverified those 39,731,200 local payload bytes against their hashes/lengths; A's bytes rely on its previously accepted full raw audit. No unchanged remote state/tensor suite was rerun. Hash equality is a raw diagnostic, not a replacement for the declared categorical rule.

The comparison authenticates A's review seal → manifest → final summary and raw-audit JSONL hashes; B's batch1 seal → JSONL/summary/vector manifest hashes; exact observation IDs, process/repeat Cartesian coverage, raw seal/canonical/driver hashes; and equality of the frozen source maps. It then consumes the complete previously raw-authenticated projections. Full matching IDs, raw record/driver/vector hashes, phase inputs, top3/spine values, and all four per-case observations are retained in `MATCHED-40-OBSERVATIONS.json`.

The unchanged accepted `q1_joint_categorical_v1.native_repeat_summary` was called with the exact policy's 84 case IDs and only the 40 available selected rows, because it deliberately rejects an invented ten-case denominator. All other 74 case rows return incomplete/unqualified, as independently asserted. Its raw **partial** output is preserved in `RAW-POLICY-PARTIAL-OUTPUT.json`; its ten completed case results do not establish full-cohort eligibility. `SUMMARY.json` explicitly leaves complete cohort admission, candidate qualification and launch authority false. No candidate was scored.

## Source and evidence identities

- Policy SHA: `4dad3bf5918c31d734214d8d2579ca1bb5b75500b575dbf2c0718287a287c593`.
- Unchanged categorical source SHA: `45187dfa879c79bbdcb2bffe963cce3ca913702628086c235bb98062ce3f40f4`.
- Accepted A review seal SHA: `2047ecc8d1aef269ede30a25507bc8bcc5a553ef50581aad17d22bf99ea04372`; manifest SHA `2b93729784a761bb009cd96ebf09da7c33f1d84e225bea1f5e00fca1a3043d41`.
- Selected A metadata is from accepted `audit-batch1.jsonl`, SHA `e04e308cbc9b84394fa55234976c6806acc87ffa3e60bd0e964e0eb3c848719a`.
- Accepted B batch1 review seal SHA: `7662cd0895eb5824369ce93d1460fc98b8a8597d38bd0a507244b7c5e046d56d`; JSONL SHA `03f64427c2df41d81f6b22adddb19e0192409e6bd824fff2a456152762320133`.

Artifacts are sealed under `p0/monitor/review-response-20260927/native-joint-common-o0-AB-first10-independent/`: exact comparison script, unchanged reducer/policy copies, matched observation evidence, CSV, partial reducer output and scope-limited summary. Review completed 2026-09-29 19:56:23 UTC using local CPU and existing evidence only. No remote operation, source/gate edit, model/GPU invocation, extra batch or complete-population data reduction/admission was performed. Stop after these ten cases.
