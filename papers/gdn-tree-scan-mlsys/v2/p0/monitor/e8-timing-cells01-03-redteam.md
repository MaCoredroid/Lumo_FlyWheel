# E8 timing cells01–03: cumulative independent review

2026-09-22. Reviewer: paper_redteam_round1. **All three sealed cells PASS / VALID; no material finding.** The remaining three cells and the final three-pair aggregate are pending. No partial primary estimate, interval or campaign gain claim is reported.

All review operations were local CPU reads/reductions. The original cell01 and cells01–02 review artifacts are preserved, along with every attempted campaign/source file.

The exact timing manifest remains `8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1`; exact qualification pass remains `815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513`. The independent checker (`e8_timing_independent_check.py`, SHA-256 `1f22d65b7145144ad44f7bcc8a836f939fe3132c44408b7675c3ed283431cf9b`) reverified all three source-bound terminal seals and all 45 sealed evidence files per cell. It independently reconstructed API/ledger binding, terminal clipping, support/exclusions, per-prefix coverage, clean head counts and ownership. Separate frozen-join executions for newly copied cells02 and03 each returned rc0 and reproduced their saved JSON exactly.

| Cell | Arm | Proposals | Primary / legacy calls | Retained API-bound tokens | Retained intervals | Retained wall seconds | Instrumented support tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 01, block1 | OFF | 360 | 1800 / 1800 | 995 | 314 | 102.7218893384561 | 9.686348317850701 |
| 02, block1 | ON | 360 | 1800 / 0 | 995 | 314 | 79.02388299256563 | 12.59113020418912 |
| 03, block2 | ON | 360 | 1800 / 0 | 995 | 314 | 78.9974324265495 | 12.595346069318577 |

These are cell-level rates on the frozen pure-decode instrumented support, not whole-service throughput. All actual loaded source/configuration gates pass. The clean ON eagle hash in both ON cells is `7a1f20b0897ec488f0a1e168105ce392cac566e79dbd206a1edeee527c0060c8`; OFF remains `959a7e9f46dc239403c8c5ed42e78e3208d80d2f3e762235c570b6ed5e837101`. The two ON cells have exactly five head evaluations per proposal and OFF has ten. All head receipts close without errors; qualification counters are zero, with no selfcheck output. Actual engine seed 20260921, Cat10 B1, TREE_ATTN flat policy 1/0/1, synchronous/eager/cache-off mode and the other pinned sources pass.

Every reviewed cell has 1,082 contiguous error-free recorder events, 360 forward/output pairs, 346 physical steps, a matching final close and no sink failure. The ten complete requests in each cell yield two 32-ID warmups plus eight 128-ID timed streams, all ending by length. All 1,088 API IDs per cell reconstruct from 1,102 structural IDs with 14 last-row-only clipped IDs (four warmup, ten timed). Four discarded empty nonpure rows contribute no tokens. Actual request-body hashes reconstructed from frozen prompt bytes match, and every request maps uniquely to the correct engine ID/phase inside the sampled workload.

The 32 excluded intervals per cell reconstruct exactly: 24 warmup intervals, seven mixed/prefill sequence gaps and one terminal interval lacking a successor. Every prefix has 34–46 retained intervals, above the frozen structural floor. None exceeds the diagnostic 1.5 s cap. The same exact per-prefix retained token/interval counts appear in all three cells; their measured wall intervals differ and remain unchanged.

Ownership samples: 30 for cell01, 24 for cell02 and 24 for cell03. All queries pass; each sample binds its sole observed GPU host PID to the exact labeled container's `docker top` host-PID set. Maximum gaps are 5.005958143621683, 5.001286894083023 and 5.0010590720921755 seconds, respectively. Every request is bracketed. All containers exited code 0 with no OOM flag. Occasional samples establish no observed contention at sample times; they cannot prove continuous isolation.

**Descriptive stream diagnostic:** all eight complete 128-ID streams match for cell01 OFF versus cell02 ON, and for cell02 ON versus cell03 ON. The prefixes are p072, p017, p015, p021, p085, p095, p058 and p083. This observation changes no retention rule and is not a full-model state-equivalence proof. No inference across an incomplete subset of the planned pairs is made.

## Evidence hashes

Cumulative JSON: `p0/monitor/e8-timing-cells01-03-independent-review.json`, SHA-256 `09f6068a7dcdf2bee857076f7013bbd35860c492c59e2f2a91a01904eb15f986`. It retains all three complete token streams, per-request clipping, per-prefix timing support, ownership summaries and source-bound terminal receipt identities.

| Artifact | SHA-256 |
|---|---|
| Cell01 terminal receipt, unchanged | 663bfff9db54293591412a298adb1cbc65141bca7f7c916d5ca489c8f6118dd8 |
| Cell02 terminal receipt | a6ada9632ae7f0777bdfbddb500ca4ac715a8b2ff1ffc7d14f99f091b847ee1f |
| Cell03 terminal receipt | bbd9eab691e4d13622fc89a5e6c2e31a223b04de5b94fc269fd87d7da1f88b5f |
| Cell02 logs/e1_events.jsonl | 78d4bc02be19a2f1af14edbe0337a092339a4ab7e47fb91a70109e40412b7c59 |
| Cell03 logs/e1_events.jsonl | 17177c6560e18df51177db5781f5cc909f43a2a49f0fdaa285aea6534ea4af9e |
| Cell02 join.json | eb8fd3e5ef462d787cde152004f165918c69cfff2948b891e6f4b16978f40c1e |
| Cell03 join.json | d5c075b9e649187a95a2f6ca0f4e44778312828523611f73b87b950ff89f31c7 |
| Both ON logs/e8_head_gate.json | 49ab824ae7bb85ce063655e742558cbddf85c2d86eb4966a0612f3a6f97482a4 |
| Cell02 ownership_samples.jsonl | 01e0c3d9cd6d042831fff42f1a63b231ecc2fb065ee6d63a7b01c0976ffea392 |
| Cell03 ownership_samples.jsonl | 7ef9b362c23dce9573d6f279e7155dd9f388e9cd566e1a258bae1a71c9662332 |

No repair, extra experiment, replacement or retention-rule change is indicated. The existing fixed campaign should continue unchanged; final assessment awaits all six sealed cells and the frozen aggregate.
