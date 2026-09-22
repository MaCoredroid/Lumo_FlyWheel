# E8 timing cells01–02: cumulative independent review

2026-09-22. Reviewer: paper_redteam_round1. **Both sealed cells PASS / VALID; no material finding.** Four planned cells remain unreviewed. The cumulative JSON contains no partial primary estimate or interval; no campaign aggregate conclusion is made.

Only local CPU reads/reductions were used. The attempted source and all original artifacts remain unchanged. Cell01's earlier report/JSON are retained.

## Evidence and validation

Run: `experiments/out-20260922T215254Z-e8-timing`. Exact timing manifest `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`; exact qualification pass `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`.

The independent checker `p0/monitor/e8_timing_independent_check.py` remains SHA-256 `1f22d65b7145144ad44f7bcc8a836f939fe3132c44408b7675c3ed283431cf9b`. It reverified both terminal seals, all 45 raw-file hashes per cell, frozen source/qualification identity and actual loaded configuration, then reconstructed complete API streams, terminal clipping, physical support, exclusions, coverage and sampled ownership. A separate fresh execution of the frozen joiner for cell02 returned rc0 and exactly reproduced its saved join JSON.

| Cell | Arm | Proposals | Primary / legacy heads | Retained API-bound tokens | Retained intervals | Retained wall seconds | Instrumented support tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 01, block1 | OFF | 360 | 1,800 / 1,800 | 995 | 314 | 102.7218893384561 | 9.686348317850701 |
| 02, block1 | ON | 360 | 1,800 / 0 | 995 | 314 | 79.02388299256563 | 12.59113020418912 |

The clean ON source is the qualified variant `7a1f20b0897ec488f0a1e168105ce392cac566e79dbd206a1edeee527c0060c8`; the actual emitted/loaded hash matches. Its closed head receipt proves five calls per completed proposal, while the OFF cell proves ten. Qualification counters remain zero and no selfcheck file exists. The actual engine seed, synchronous/eager/cache-off settings, Cat10 B1 occupancy, flat TREE_ATTN KV policy 1/0/1 and pinned other loaded modules all pass.

Each cell has 1,082 contiguous error-free recorder events, 360 forward/output pairs, 346 physical steps and a matching clean close. Each also has ten complete API requests: two 32-ID warmups plus eight 128-ID timed requests, all terminating by length. All 1,088 API IDs reconstruct from 1,102 structural IDs with 14 terminally clipped IDs (four warmup, ten timed); four empty discarded nonpure rows add no tokens. Every request body hash, engine ID mapping, phase and sequential submission boundary passes.

The same 32 exclusion decisions are independently reproduced in each cell: 24 warmup intervals, seven mixed/prefill sequence gaps and one terminal interval lacking a successor. All eight timed prefixes have 34–46 retained intervals, exceeding the frozen structural support floor. No retained interval is above the diagnostic 1.5 s cap. These rates describe the frozen pure-decode instrumented support, not whole-service throughput.

Cell02 has 24 passing ownership samples that bracket every request in the complete workload; the largest recorded gap is 5.001286894083023 seconds. Its sole sampled GPU host PID 3605650 belongs to the exact labeled container's `docker top` host-PID set. Cell01's 30 samples still verify. No observed foreign process/container or failed query appears; sampled evidence cannot exclude activity wholly between observations. Both containers exited with code 0 and no OOM flag.

**Descriptive stream comparison:** all eight complete 128-ID ON/OFF timed streams match exactly: p072, p017, p015, p021, p085, p095, p058 and p083. Equality is not used to retain or exclude either cell and does not prove arbitrary full-model state equivalence. No estimate across a subset of the three planned pairs is reported.

## Hash ledger

Cumulative JSON: `p0/monitor/e8-timing-cells01-02-independent-review.json`, SHA-256 `dadf09e5ed8ac05143ee4bdd08e036f856a0c7074dc548156ac9e51e69d3201a`.

| Artifact | SHA-256 |
|---|---|
| Cell01 terminal receipt, unchanged | 663bfff9db54293591412a298adb1cbc65141bca7f7c916d5ca489c8f6118dd8 |
| Cell02 terminal receipt | a6ada9632ae7f0777bdfbddb500ca4ac715a8b2ff1ffc7d14f99f091b847ee1f |
| Cell02 logs/e1_events.jsonl | 78d4bc02be19a2f1af14edbe0337a092339a4ab7e47fb91a70109e40412b7c59 |
| Cell02 join.json | eb8fd3e5ef462d787cde152004f165918c69cfff2948b891e6f4b16978f40c1e |
| Cell02 logs/e8_head_gate.json | 49ab824ae7bb85ce063655e742558cbddf85c2d86eb4966a0612f3a6f97482a4 |
| Cell02 ownership_samples.jsonl | 01e0c3d9cd6d042831fff42f1a63b231ecc2fb065ee6d63a7b01c0976ffea392 |
| Cell02 docker_inspect.json | 260d5cb56b45ddcfac0c26c678dfc86f635fecb80f308d2b733160d962970070 |

No repair, additional experiment, replacement or retention-rule change is indicated. Continue the existing frozen schedule and wait for all six sealed cells and the fixed aggregate before making an E8 campaign claim.
