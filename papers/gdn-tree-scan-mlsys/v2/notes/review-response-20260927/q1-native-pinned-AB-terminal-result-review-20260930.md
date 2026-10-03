# Pinned native A/B terminal result: independent review

The original frozen reducer completed at 07:02:05 UTC on 30 September 2026. Its complete result is authenticated and adverse: all336 observations are valid and accounted for, but the joint native reference prerequisite holds for only68/84 cases. No audit-reuse wrapper was created or run; the earlier reuse discussion remained operational analysis.

## Exact result and counts

- Complete reduction: `ccac60adccc1457a0db90ac79666d4f8972cceef0e7ed78d6f9195b17b3360f1` (`COMPLETE-AB-REDUCTION.json`).
- Frozen retry1 corpus: `8b1ab3e055292801e1e40b3f6372b49bd48377f30d60f6eb5df01e74115159ae`.
- Accepted reducer: `d78183cee56379a20ebb83628e8b8cfd9e286b04a162884cbc0fa191295955f5`.
- Unchanged categorical contract: `4dad3bf5918c31d734214d8d2579ca1bb5b75500b575dbf2c0718287a287c593`.

| Property across A/B × R2 | Cases /84 |
|---|---:|
| Exact target starting state |84|
| Exact published target state |65|
| Exact full target logits |65|
| Stable target smallest-ID greedy |84|
| Stable MTP before-z first spine/ordered-top3 |75|
| Stable MTP before-z follow spine/ordered-top3 |77|
| Stable MTP after-z first spine/ordered-top3 |76|
| Joint categorical native prerequisite |68|

All four phases have matched scalar token/position/extent/dtype/operator inputs for all84 cases. Each phase has identical full-score bytes across all four observations for65/84 cases. Within each process, both repeats' phase score hashes match on every case. The16 adverse joint cases are all in the long-prefix block: n01,n04,n05,n10,n11,n14,n15,n16,n19,n20,n21,n24,n28,n29,n30,n31. Failed-case and phase membership is retained in the JSON seal; none is dropped from the denominator. Equal target greedy decisions do not override the MTP ordered-top3 criterion.

## Independent reconciliation

I reconciled every one of336 unique case/process/repeat rows with the complete independently audited A and B JSONL logs, including exact dictionary equality with the completed reducer rows. Their full byte hashes and168-row progress receipts are bound in `REVIEW-SEAL.json`. I rechecked all336 actual case-file hashes and336 driver-file hashes against both those logs and the terminal inventories, plus22 selected terminal/config/boot/cache members:694 indexed metadata hashes in total. This does not repeat the accepted reducer's full state/object arithmetic.

Both terminal receipts report `COMPLETED_driver_rc=0_cleanup_rc=0`, driver exit0 and stopped/removed owned engines. A receipt SHA is `b617bd6b277af94713205750bfef4065801ed6c599d5970194f27e2528fa6188`; B is `ba3de16dfc95a16b4568fb0003be5802b3acf72bed8cb3d20dbb33cb6822e24e`. The actual distinct container IDs, exact immutable image `ffa30d66…`, image-plus-arm environment, entrypoint/command, mounts,105GiB memory/swap policy, zero restart count, spec-off aligned route,48 GDN/16 FA2 layers, and patched FA2 binary were checked against the recorded configuration and launch bindings. The existing complete reducer additionally authenticates the full gate, patch, source-manifest and all terminal member conjunctions.

Both processes loaded and retained the same65 exact kernel-selection entries as original cache SHA `dfc398ac0f3d14fe3d1cbbb6d4b277ecc3971c27e42470e6a5675981a1621e5a`. The actual runtime export entries were compared with each other and that original cache; no new tuning entries replaced it. This validates the pin's recorded effect, not determinism of every possible operation.

I independently read all96 adverse phase full-vocabulary vectors:248320 FP32-exported values each, 48 unique hashes. Raw length/SHA, finiteness, smallest-ID greedy, captured exact-k IDs and their ordered score values matched. Every exported value is exactly BF16-representable, consistent with the recorded native head dtype. Strict-boundary membership changes and unequal-score order changes exist, so tie handling alone cannot explain away the failure. These are96 bounded score reads, not96 experiments or a second complete cache scan. The accepted original reducer reports106320 target raw objects /26599645184 bytes checked per process, with its additional joint/source audits retained separately.

Thirty-six small result/progress/log/terminal/config metadata files were copied locally and checked; no multi-GB case/raw payload cohort was copied or rescanned. The read-only audit source, raw JSON reconciliation, metadata verification manifest and final seal are under `p0/monitor/review-response-20260927/native-retry1-AB-independent-final-20260930/`.

## Paper-result wrapper and interpretation

`build_native_joint_pinned_paper_result_v1.py` SHA `0954b8ad947a20577bef6f1484d23c6bcc931fe5189f943d5f5d440ac04988f0` verifies unchanged base SHA `92d14b5d56867eb68d685830acdf05b09021992194a7208ebdcd3809d68898be`, then overrides both corpus path and SHA to the retry1 corpus above. The base derives counts at call time from those globals, reconstructs the336/84 population and all phase/count conjunctions, and refuses a reduction bound to old corpus `ac3f1248…`. Only the rendered introduction changes to identify the kernel-pinned follow-up. The wrapper still requires parent review receipt, independent note and seal; this review did not generate a parent admission or edit the manuscript.

Disposition: complete, reviewed native-reference repeatability result under the unchanged rule; joint prerequisite **not qualified**. The raw failures concern the native end-to-end route from common imported target/MTP state. Changed downstream hidden/state operands prevent attributing them specifically to an isolated MTP kernel on identical operands. They establish neither candidate failure nor algorithmic impossibility. Candidate/full-Q1/continuous/lifecycle/held-out/timing/WP claims remain unqualified. Supported claim narrowing may close campaign work as documented in the prior disposition review; it does not turn these findings into a PASS or new workload evidence.
