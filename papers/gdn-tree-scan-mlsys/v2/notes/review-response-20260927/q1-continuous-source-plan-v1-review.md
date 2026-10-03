# Continuous calibration source plan v1: independent review

2026-09-29. **PASS for the repaired, source-only preparation glue.** No launch, model/GPU/container/remote operation, qualification, new workload attempt, or criterion change. The original defect and failing controls are preserved.

## Reviewed bytes and scope

- Current `tools/q1_continuous_source_plan_v1.py`: `5e1dfe7c4c40bbd3665045ad394091a84c0e30c3cefc4332b9362503c68a40df`.
- Original source: `df4528179fcd14131206a3326966a00abf1c14516294b43ad3f5ff16eaf311b6`.
- Parent test source: `5af0cdf86bb9533cbb8ff517faea06ad7d2ccfba374e0a3cd7523598fba20883`.
- Accepted inventory/schedule dependencies remain unchanged: `7c1fca833b050b74bc34ff971b0ee09da59b5b60752fba273ae9283391f7c0a0` / `c7c1e53bc2cb2be78f9ef143f6eacd804c2c50765b23f9798fca142c1755afb1`.
- Frozen input fixture: `e9683d2f8097ecfe195331d9977f9e57e1ab8482f1d583e04c5268ad92777318`; accepted three-source inventory: `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8`.

This tool prepares nine missing **calibration prefix/root source inputs**, each from the first fixed representative and its first cycle. It does not add qualification repeats or candidate observations. The three exact existing calibration source groups are omitted from the new request list and remain reusable references subject to their existing raw authentication. Held-out input identities are read by the accepted inventory dependency, but no held-out source is placed in the executable projection, no held-out output is consumed, and evaluation execution stays closed.

## Exact independent stream reconstruction

The independent control groups the frozen calibration records by complete padded byte strings and root token, rather than calling the implementation's inventory to construct its expected answer. It obtains twelve calibration groups: three reusable and nine new. All nine new groups contain one fixed F4 record; all projected cases retain `F1-n31-00000000000`, accepted length 11, and the exact 12 materialized tokens plus pending Z (13 tokens). All prefixes, token sequences, positions, record mapping, prefix digests and extents match. The full token/position arrays are retained in `results.repaired.json`; the following is a compact census.

| Fixed calibration representative | Prefix length | Root token | Projected positions |
|---|---:|---:|---:|
| short_available__F4-B3-kv-block-boundary-minus1 | 14335 | 11352 | 14335–14347 |
| short_available__F4-B3-kv-block-boundary-minus3 | 14333 | 11352 | 14333–14345 |
| short_available__F4-B3-kv-block-boundary-minus11 | 14325 | 11352 | 14325–14337 |
| medium_available__F4-B3-kv-block-boundary-minus1 | 29695 | 363 | 29695–29707 |
| medium_available__F4-B3-kv-block-boundary-minus3 | 29693 | 363 | 29693–29705 |
| medium_available__F4-B3-kv-block-boundary-minus11 | 29685 | 363 | 29685–29697 |
| long_available__F4-B3-kv-block-boundary-minus1 | 60415 | 7643 | 60415–60427 |
| long_available__F4-B3-kv-block-boundary-minus3 | 60413 | 7643 | 60413–60425 |
| long_available__F4-B3-kv-block-boundary-minus11 | 60405 | 7643 | 60405–60417 |

Selection is bound to fixture SHA and accepted-source SHA through `Inventory.build` (new source lines 33–44); it never ranks outputs. Original prefix bytes are rechecked, only frozen uint32 padding is appended, and full padded hashes/extents are required (lines 46–52). `job` regenerates and compares the entire plan and exact requested run, compares the prepared projection, and checks all nine prepared byte payloads before invoking the unchanged `ReferenceJob.build` (lines 85–102). The result retains one aligned-nonpacked process A, one repeat and nine requests; state/KV byte archival remains enabled. Projection JSON may be reformatted without changing its content; the actual serialized fixture hash is separately recorded by the accepted job builder.

## F1 preserved and closed: mutable native-policy alias

Original lines 80 and 97 returned the imported `MTP_POLICY` dictionary directly. With the real nine-source build, mutating `plan['native_mtp_policy']['full_raw_history']=False` also mutated the expected authority: `job()` regenerated that altered expected plan and accepted/emitted `full_raw_history=False`. Mutating a returned job's policy also changed future builds. Both controls fail on the preserved original source; ordinary detached plan mutations were correctly refused.

The successor adds only `import copy` and a `copy.deepcopy(MTP_POLICY)` at each output boundary (current lines 81 and 98). The exact original two counterexamples now refuse or preserve future policy identity respectively. No token, selection, count, criterion, or accepted dependency changed.

## Executed CPU controls

**35/35 independent controls pass** on the successor; the same script produces **33/35** on the original, failing exactly the two policy-alias checks. Controls cover independent nine-stream reconstruction, complete legacy job scope, omitted/duplicated/reordered sources, held-out activation, repeat/process changes, foreign representative, stale inventory/accepted-source pins, detached policy change, foreign run, each of nine corrupted prefix payloads, changed prepared positions and held-out flags, and both live-object alias cases. The two parent unittest methods also pass independently on the successor.

Commands from repository root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/continuous-source-plan-review/controls.py
PYTHONDONTWRITEBYTECODE=1 python3 papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/continuous-source-plan-review/controls.py papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/continuous-source-plan-review/source.repaired.py papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/continuous-source-plan-review/results.repaired.json
PYTHONDONTWRITEBYTECODE=1 python3 -W ignore::ResourceWarning -m unittest discover -s papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests -p test_q1_continuous_source_plan_v1.py -v
```

The reviewer's first harness attempt used the wrong ancestor directory and failed at module import before executing reviewed code; the erroneous harness and explanation are retained. This was an auditor setup error, not an implementation defect.

## Boundaries still pending

No launcher selects this plan. A future owned launcher must bind this exact source and its dependencies/prepared files, pass the prepared prefix directory to the accepted driver, and provide its existing run/gate/source/runtime authentication. The new plan does not itself inspect GPU state or authenticate reusable tensor records. These are explicit unconnected admission boundaries, not additional experiments or a failure of this CPU source preparation. This review does not qualify continuous maps, native/candidate numerical behavior, full-model state, performance, or workload execution.
