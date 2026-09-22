# E8 timing cell01: independent sealed-data review

Date: 2026-09-22. Reviewer: paper_redteam_round1. **PASS / VALID for cell01 OFF. No material finding.** Five cells and the fixed three-pair aggregate remain pending; this cell alone supports no ON/OFF gain claim.

Run `experiments/out-20260922T215254Z-e8-timing/cell_01_block1_off`. Reviewed locally with CPU-only reads/reductions; no DGX, Docker, GPU or inference operation. No attempted source or campaign file was changed.

- Timing manifest: `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`.
- Qualification pass: `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`.
- Terminal cell receipt: `663bfff9db54293591412a298adb1cbc65141bca7f7c916d5ca489c8f6118dd8`; all 45 sealed evidence file hashes and the exact source snapshot verified.
- Independent checker: `p0/monitor/e8_timing_independent_check.py`, SHA-256 `1f22d65b7145144ad44f7bcc8a836f939fe3132c44408b7675c3ed283431cf9b`.
- Independent result JSON: `p0/monitor/e8-timing-cell01-independent-review.json`, SHA-256 `a1c6e723af9c1006c67aa3f5ab7e2d8dabf9b51eb7fd9c8f2cddd2f750cf21fd`.

## Raw checks and result

The clean OFF source hash is `959a7e9f46dc239403c8c5ed42e78e3208d80d2f3e762235c570b6ed5e837101`. The emitted shim, actual loaded eagle and frozen clean variant match. The other loaded modules and six frozen repository overlays match their pinned sources. Actual mounts make `/e8`, repository overlays and model files read-only. Actual engine/route checks pass: seed 20260921, B1, root plus nine drafts, TREE_ATTN, KV policy 1/0/1, synchronous/eager/cache off, runrow/replay/eager-pack/fused-conv enabled. Selfcheck and qualification dispatch/comparison counters are inactive; no selfcheck output exists.

The head receipt closes without failure with **360 completed proposals, 1,800 primary calls and 1,800 legacy calls: exactly 10 head evaluations per proposal**. Head and recorder owner PID 192 agree. This container PID is not confused with the independently observed host GPU PID.

All ten API request bodies were reconstructed from the frozen prompt text and settings and matched their captured SHA-256. Two warmups returned 32 IDs each and eight timed requests returned 128 each, all with `length` termination. Every request maps uniquely to its engine ID; sequential submission and the ownership-window bounds pass. Complete raw output-row concatenations reconstruct all **1,088 API IDs** from **1,102 structural IDs**. The 14 clipped IDs occur only in terminal rows: four warmup and ten timed. Four discarded empty nonpure prefill rows contribute no tokens.

The raw recorder has 1,082 contiguous, error-free events, 360 forward/output pairs and 346 physical steps. The final close matches those totals with no sink failure. Independently rebuilding consecutive same-request pure intervals gives **314 retained intervals, 995 API-bound tokens and 102.7218893384561 seconds**, yielding **9.686348317850701 tokens/s** on the frozen instrumented pure-decode support. This is not complete service throughput. All 32 exclusions match exactly: 24 warmup intervals, seven explicit mixed/prefill sequence breaks, and one terminal interval without a successor. No retained interval exceeds the diagnostic 1.5 s cap. Re-executing the frozen joiner in a temporary directory produces the exact saved JSON with rc 0.

| Prefix | Retained intervals | API-bound retained tokens | Retained wall seconds |
|---|---:|---:|---:|
| p072 | 39 | 125 | 12.690085510 |
| p017 | 40 | 123 | 12.991149505 |
| p015 | 40 | 123 | 12.984212524 |
| p021 | 38 | 124 | 12.339304008 |
| p085 | 46 | 126 | 15.130293978 |
| p095 | 42 | 123 | 13.830930165 |
| p058 | 35 | 126 | 11.557846464 |
| p083 | 34 | 125 | 11.198067185 |

Every prefix exceeds the frozen one-interval minimum; total support exceeds eight. Thirty raw ownership samples bracket the complete warmup/timed workload and contain only the exact labeled container. GPU host PID 3599828 belongs to its `docker top` host-PID set at every sample. All queries succeeded; the largest recorded gap is 5.005958143621683 seconds. This establishes no observed sampled contention, not continuous isolation between observations. The final container exit is 0 with no OOM flag.

## Additional evidence hashes

| File | SHA-256 |
|---|---|
| logs/e1_events.jsonl | 00da04a6dacab5ce68d6d02ff56d801022554f0ee68de908c498d5b861d40d68 |
| join.json | f4411d9c9f0d1e07999eac24bc51e495a0de30c4e4d4803c06dcb36c0a7b013d |
| logs/e8_head_gate.json | d3f561749ef0815e6b718b1811a7bc334c9f84d80ae4520d1024f1369b2b8fed |
| ownership_samples.jsonl | 85dedf56a6d4edbb7ab9ea4c1e90a40e8c6893302b4676997e5a69243261463e |
| docker_inspect.json | 90eb6dd6f565da3a01723949082cb4f2586d64a051c69b6cb9a280f01c52c465 |

The first and last retained examples are physical step 24/forward 28 (0.3228046949952841 s) and step 344/forward 359 (0.333853367716074 s); both bind one timed request and the same-request successor. The independent JSON retains per-prefix support, complete streams, each request's clipping, ownership summary and terminal receipt identity. The original infrastructure failure and qualification evidence remain preserved. Continue only the existing frozen schedule; no additional experiment, retry, replacement or change to the retention rule is indicated by this review.
