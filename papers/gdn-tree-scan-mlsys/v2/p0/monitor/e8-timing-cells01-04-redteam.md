# E8 timing cells01–04: cumulative independent review

2026-09-22. **All four sealed cells PASS / VALID; no material finding.** Two planned cells and the final three-pair aggregate remain pending. No partial aggregate or confidence interval was calculated. Earlier reviews, all campaign artifacts and the attempted source snapshot are preserved unchanged.

Reviewed only local copied evidence with CPU reductions. Exact timing manifest: `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`. Qualification pass: `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`. Independent checker remains `e8_timing_independent_check.py` SHA-256 `1f22d65b7145144ad44f7bcc8a836f939fe3132c44408b7675c3ed283431cf9b`.

The cumulative checker reverified every source-bound seal and all 45 evidence hashes in each cell, then reconstructed the complete warmup/timed API streams, physical support, exclusions, coverage, head census and ownership. The newly copied cell04's frozen joiner was separately executed in a temporary directory: rc0, exact match to its saved join JSON.

| Cell | Arm | Retained API-bound tokens | Intervals | Retained wall seconds | Instrumented support tokens/s | Primary / legacy head calls |
|---|---|---:|---:|---:|---:|---:|
| 01, block 1 | OFF | 995 | 314 | 102.7218893384561 | 9.686348317850701 | 1800 / 1800 |
| 02, block 1 | ON | 995 | 314 | 79.02388299256563 | 12.59113020418912 | 1800 / 0 |
| 03, block 2 | ON | 995 | 314 | 78.9974324265495 | 12.595346069318577 | 1800 / 0 |
| 04, block 2 | OFF | 995 | 314 | 102.54264487046748 | 9.70328004760254 | 1800 / 1800 |

All cells have 360 completed proposals: clean ON has five head calls per proposal and OFF has ten. Cell04's actual loaded eagle hash is the exact clean OFF variant `959a7e9f46dc239403c8c5ed42e78e3208d80d2f3e762235c570b6ed5e837101`. Its other source/module identities and actual engine/route controls pass, including explicit engine/API seed 20260921, Cat10 B1, synchronous/eager/cache off and flat TREE_ATTN KV policy 1/0/1. Qualification counters remain zero and no selfcheck output exists.

Each reviewed cell independently reconstructs ten complete requests: two 32-ID warmups and eight 128-ID timed requests, all ending by length. The 1,088 API IDs bind to 1,102 structural IDs with 14 last-row-only clips. All actual request-body hashes match the frozen prompt bytes/settings and each API response maps uniquely to the correct engine request. Four discarded empty nonpure rows add zero tokens.

Each ledger contains 1,082 contiguous error-free events, 360 forward/output pairs and 346 physical steps, with matching close totals and no sink failure. Each has the same 32 explicit interval exclusions: 24 warmup, seven mixed/prefill sequence gaps and one terminal interval lacking a successor. The 314 retained intervals and 995 API-bound tokens above exactly match independent reconstruction. All eight prefixes have 34–46 retained intervals; none exceeds the diagnostic 1.5 s cap. These rates cover the frozen instrumented pure-decode support, not complete service throughput.

Cell04 has 29 passing ownership samples, host GPU PID 3615373 bound to the exact labeled container's host-PID set, and maximum observed gap 5.001365535892546 seconds. Its samples bracket all requests. Cells01–03 ownership evidence also revalidates. All four containers exited code 0 without an OOM flag. This proves no observed sampled contention, not continuous absence of activity between samples.

Descriptively, every complete 128-ID prefix stream in block2 ON cell03 exactly matches OFF cell04 (8/8). The earlier block1 comparison also remains 8/8. Equality changes no retention rule and is not a full-model state-equivalence claim. No inference is made from an incomplete subset of the planned three pairs.

## Exact review evidence

Cumulative JSON: `p0/monitor/e8-timing-cells01-04-independent-review.json`, SHA-256 `234f496ca133ee716919249f3709d19734c74cb136222a0ff753ba8103c8dc7f`.

| Cell04 artifact | SHA-256 |
|---|---|
| cell_result.json | 8775180416812662aa4aa4c67939baa27a7eb8789e9f94cda29929bd9db4f007 |
| logs/e1_events.jsonl | 80222e219d82aa4851d3ed5f90e2253219fc48a4627ad3e296072e76bf8b320c |
| join.json | 8be4e1db3adb4bce17cf3553e3897c3dccd04e021b03a1acb3afebdc205dd776 |
| ownership_samples.jsonl | 9c1c8e893064a3d2d85eac9525d61851e898c98481de4d7531d88bc88addb852 |
| logs/e8_head_gate.json | d3f561749ef0815e6b718b1811a7bc334c9f84d80ae4520d1024f1369b2b8fed |
| docker_inspect.json | a751df2371e72b10fd9ef25ee337d73d0f21c679b8be2d715412718c614bbe7b |

The JSON retains all four terminal-receipt hashes and complete reconstructed evidence. No repair, extra experiment, favorable replacement or retention-rule change is indicated; continue the existing frozen schedule.
