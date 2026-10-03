# Component-qualification manuscript integration review

Reviewed 2026-09-28 UTC / September 27 Pacific. **Bounded PASS: no material integration issue found.** This covers only the new generated GDN component subsection, its adjacent JSON and generator, and its insertion in the numerical-validation section. It is not a whole-paper review, new numerical reduction, performance qualification or full-model approval. No author/production code, tests, GPU workload or generator was executed, and no manuscript/evidence file was edited.

## Verified claims

- All **six input hashes** recorded in `results/review-response-20260927/component-qualification.json` match their actual local files. Both accepted summaries have 5,760 PASS cases in each process, zero malformed/structural/numerical/uncovered findings, and independently identified processes. The calibration and held-out acceptance receipts and preceding independent raw reviews agree.
- The table denominator is **per block and process**: two fixtures × 48 recurrent instances × 32 physical output rows = 3,072 output cases; two × 48 × 28 accepted paths = 2,688 publication cases. Their sum is 5,760. Each case contains 48 heads; it is not one scalar/head observation. The text correctly includes padding in the output denominator and states that repeats/processes add reproducibility checks, not distinct-input coverage.
- Calibration output identity is **3,004/3,072**, with 68 differing stored BF16 tensors; held-out is **3,018/3,072**, with 54 differences. Both publication counts are **2,688/2,688** bitwise native. Prior raw audits establish candidate/native repeat and cross-process identities. The subsection does not convert numerical PASS into bitwise output equality.
- The displayed paired formula agrees with policy SHA `4a103df013439ae02977d10a42c3cf2eeb916a7dcb1a61c19e6b929da620f7b3`: kappa=1.1, FP32 unit roundoff `2^-24`, and `2^-149`, applied separately to RMS and maximum absolute error for every required head/case. Reference magnitude uses the matching RMS or maximum metric. No extra BF16 allowance is introduced. The finite fixture domain and pre-candidate freeze are supported by the policy and accepted execution chronology.
- Calibration controls are described at their demonstrated scope: sibling, ring and stale-metadata mutations detected; tested off-path poison leaves the selected state unchanged; structural zero-replay control rejected. The text correctly assigns these controls to calibration, rather than implying they were repeated during held-out execution.
- Current-route source/flags, actual scan outputs, captured publication and native running-row reads support the stated component mechanism, as documented in the independent calibration/raw reviews. The wording does not claim a single fused all-layer kernel or import older-route proof. The new text explicitly leaves convolution/KV continuation and next-forward decisions to full-model qualification.
- The held-out result is the accepted **separate CPU re-reduction** of unchanged tensors under unchanged numerical policy. The adjacent JSON retains that provenance; original metadata-identity refusal remains preserved in the accepted audit. The paper does not represent this as a fresh second GPU dataset or enlarge its coverage.

The generator was inspected as source. Its fixed expected counts agree with the independently audited census, and it binds the numerical summaries and output-difference audits before constructing the rows. No further experiment or source change is necessary for these bounded claims. M1 mechanism comparison, full-model qualification and application-level conclusions remain separate.

## Reviewed hashes

Paths below are relative to paper v2.

| File | SHA-256 |
|---|---|
| `main.tex` (insertion at lines 298–306) | `fb1571cf1dd4fb67d71d3122dc63dc9f1928dc2b026795f85cd43b1939c40013` |
| `results/review-response-20260927/component-qualification.tex` | `dbdb81ffb279923c2cc7ace1cbf5bc6cb11c3921338457e8dbd85c350979c57d` |
| `results/review-response-20260927/component-qualification.json` | `a97661e2756c62e356d16c01183d312d661f465ed2886612e07c537aa22f1ea9` |
| `scripts/build_component_qualification.py` | `9dd34dd50e86087a057efaace0b74610e9465692eddc28a7eb6ca157d63f1f14` |
| `p0/monitor/review-response-20260927/q12b-calibration-result-acceptance.json` | `9d56a48cd15bc981536a2059451af0d638fc39da899b892636ce7e2d5e3963cc` |
| `p0/monitor/review-response-20260927/q12b-heldout-result-acceptance.json` | `f0107abc93a6b6aa0d46b8317c2b184612c91d7b3083ed39ebaebe577a115b7f` |
| Calibration `summary.v2.json` | `68408fc1f530ca7f8de37eb4689f5becc541adfea0b2a21905838ce60e1141f6` |
| Held-out `summary.v2_2_1.json` | `a1a52aaf0ab290647865873e805646dcf3d87f5e3480132da8c6489c31951a71` |
| `notes/review-response-20260927/q1-2b-calibration-retry-review.md` | `96e94bfd09a718b0337569a8aa69e9b825d16fd92b141577956921e88f4f627c` |
| `notes/review-response-20260927/q1-2b-heldout-rereduction-result-review.md` | `4cf4881f9e7cd8a36671cd0c92837492e87f6f285cf0a63141eb4caeaec78d0f` |

Exact summary and supplemental output-difference paths/hashes are already recorded in the reviewed generated JSON. This review rechecked those small-file bindings and read the prior raw-audit reports; it did not repeat tensor-level computation or claim new independent tensor verification.
