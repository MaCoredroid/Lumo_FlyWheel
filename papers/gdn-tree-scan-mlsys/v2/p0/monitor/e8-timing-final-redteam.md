# E8 final timing campaign: independent actual-data review

2026-09-22. Reviewer: paper_redteam_round1. **PASS: all six fixed cells are VALID, the complete frozen aggregate reproduces, and no material actual-data finding remains unresolved.** No additional experiment or replacement is needed for the bounded as-executed instrumented comparison. This is not a model-quality, cross-boot output-equivalence or universal full-model state-equivalence result.

All operations were local CPU reads/reductions. Original attempts, source freezes, raw data and earlier reviews were preserved unchanged. The preserved first infrastructure attempt failed before model/container execution; it is not a substituted timing measurement.

## Exact campaign and terminal binding

Run: `experiments/out-20260922T215254Z-e8-timing`.

| Artifact | SHA-256 |
|---|---|
| Timing manifest | 8bbb5b486fa9e11ff0bc0ef6d0373de65bbd8217392e973a02de015df39c2ab1 |
| Qualification source manifest | bc2feab326cb67c916ea418b0399499f4986e18d6bba3e4efc7a20a50f0ce030 |
| Actual two-arm qualification pass | 815b348bdb95c455122e7f1232449d4e12dd339edbd6c3c8e2b9a980c6342513 |
| CAMPAIGN_STARTED.json | e57b2e240c7986b697cf8ef447ac4f09d45db6cbeab1250f5ad6395e59a09895 |
| CAMPAIGN_COMPLETE.json | 443ab23dae8912ba457eb516f07d51f6c46ed89a65472758a99ed677943f454e |
| aggregate.json | 74b0988b191bb68fad43d41f76f520d49d529e86753ac0f18535721f6c3757fb |
| Independent checker | 1f22d65b7145144ad44f7bcc8a836f939fe3132c44408b7675c3ed283431cf9b |
| Independent final review JSON | 274566f0a0efec3c108931a329a54cc514bd0c5e783e7ec72415210669574544 |

The checker is `p0/monitor/e8_timing_independent_check.py`; its result is `p0/monitor/e8-timing-final-independent-review.json`. It verified all six terminal identities and all 45 sealed raw-file hashes per cell, all source-manifest payloads, exact qualification receipts and the campaign's final aggregate hash. No cell is missing, failed, replaced or excluded for support. The start record retains the frozen OFF/ON, ON/OFF, OFF/ON order. The terminal marker was written by the immutable driver after aggregate completion. All six owned containers exited code 0 without OOM; no assertion about a still-unseen outer wrapper receipt is needed for this data verdict.

## Independently reconstructed observations

| Cell | Arm | Proposals | Primary / legacy head calls | Retained API-bound tokens | Retained intervals | Retained wall seconds | Instrumented support tokens/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 01, block 1 | OFF | 360 | 1800 / 1800 | 995 | 314 | 102.7218893384561 | 9.686348317850701 |
| 02, block 1 | ON | 360 | 1800 / 0 | 995 | 314 | 79.02388299256563 | 12.59113020418912 |
| 03, block 2 | ON | 360 | 1800 / 0 | 995 | 314 | 78.9974324265495 | 12.595346069318577 |
| 04, block 2 | OFF | 360 | 1800 / 1800 | 995 | 314 | 102.54264487046748 | 9.70328004760254 |
| 05, block 3 | OFF | 360 | 1800 / 1800 | 995 | 314 | 102.6132529862225 | 9.696603226618251 |
| 06, block 3 | ON | 342 | 1710 / 0 | 965 | 298 | 74.88110671844333 | 12.887095854878957 |

Every clean ON proposal executes five counted heads; every OFF proposal executes ten. The actual loaded eagle hashes match the qualified clean variants (ON `7a1f20b0897ec488f0a1e168105ce392cac566e79dbd206a1edeee527c0060c8`, OFF `959a7e9f46dc239403c8c5ed42e78e3208d80d2f3e762235c570b6ed5e837101`). The other loaded modules, source overlays, image/model identity and actual settings all pass: explicit engine/API seed 20260921, Cat10 B1, stock TREE_ATTN flat KV policy 1/0/1, synchronous/eager/cache off, unchanged candidate/continuation mechanism. Selfcheck and paired-logit/dispatch qualification instrumentation are inactive; head receipts close without failure and bind the recorder owner.

The raw checker independently reconstructed the actual request bodies from frozen prompt bytes and settings, unique API-to-engine identities, sequential warmup/timed phases, complete emitted-token sequences and last-row-only clipping. All 60 API requests are accounted for, including 12 warmups and 48 timed requests. The entire campaign has 6,501 API IDs (6,117 timed) and 85 clipped structural IDs; these clips never enter the retained numerator. Cell06 p021 ends by EOS/stop at 101 API tokens; all other timed requests have 128 IDs. This prespecified early stop remains included without replacement.

Raw event counters, forward/output sequences, request row indices, physical IDs and closing totals reconcile in every cell. Independent interval reconstruction uses consecutive pure same-request physical starts, exact phase exclusions and API-bound numerator clipping. It matches every retained row and exclusion in the saved join results. Totals are 1,868 retained intervals and 5,940 retained API-bound tokens. Per-cell support is 298–314 intervals; every timed prefix has 29–46 intervals, exceeding the frozen floor. All valid slow intervals would be retained; none exceeds the diagnostic 1.5 s cap in this campaign. Each of cells01–05 excludes 24 warmup intervals, seven explicit mixed/prefill gaps and one terminal interval; cell06 excludes 22 warmup intervals, seven such gaps and one terminal interval.

The frozen joiner was independently replayed for each newly reviewed cell as it sealed, including final cells05/06 in this pass. Each produced the exact original full JSON and rc 0. The final frozen aggregate was also rerun into a temporary directory. It matches the entire archived aggregate after normalizing only the new absolute `qualification.root` metadata path for the local copy; all numeric values, cell records, stream diagnostics, source hashes and original archived bytes match unchanged.

There are 160 passing ownership samples (30/24/24/29/30/23 across cells). Each raw query set succeeds and binds all observed GPU host PIDs to the exact labeled container's `docker top` host-PID set, using host rather than container PID numbers. Samples bracket the complete warmup/timed workloads and every request. Maximum observed gap is 5.005958143621683 seconds. This is evidence of no observed contention at sampled times, not continuous isolation proof.

## Frozen primary analysis

The three paired relative differences `(ON−OFF)/OFF` are **29.9884104001%, 29.8050350761%, 32.9031987150%**. Their arithmetic mean is **30.8988813971%**. Independently replaying all 10,000 seeded paired-block resamples and the specified interpolation rule gives the frozen 95% percentile interval **[29.8050350761%, 32.9031987150%]**. The resampling unit is the paired block; with only three blocks this is coarse uncertainty, not thousands of independent experimental replications.

The diagnostic ratio of arm means minus one is 30.8989513058%; it is not substituted for the frozen primary. Cell-rate means are 12.691190709462218 tokens/s ON and 9.695410530690497 tokens/s OFF. The primary was frozen before timing and includes all three pairs, regardless of sign, precision or output-stream behavior. No subset or favorable retry was used.

## Required interpretation limits

**Stream equality is not universal:** 16/24 within-pair across-arm prefix comparisons and 32/48 within-arm across-block comparisons match exactly, for 48/72 overall. The first five cells have matching complete streams; all eight cell06 ON streams differ from their block3 OFF counterparts and earlier ON repetitions. The 101-token EOS case is part of this difference. These are descriptive diagnostics and did not affect retention or the estimate. The measured gain therefore describes this source-controlled configuration comparison on the fixed workload, with observed continuation differences; do not label it identical-output service acceleration or a model-quality result.

The prior actual qualification provides same-input in-memory full-logit, ordered-candidate, hidden/RNG and mutation checks for its tested proposals. The clean timing census establishes the executed head-count difference. Neither this qualification nor the timing campaign supplies archived full-vocabulary tensors for an offline equality replay, arbitrary cross-boot state equivalence, or isolated LM-head kernel timing. Report the observed gain on the instrumented pure-decode support with these limits. B4, other backends, graph/cache/stochastic modes and composed optimizations are outside this result. No extra experiment is indispensable for that bounded claim.

## Final cell receipt hashes

- `cell_01_block1_off/cell_result.json`: `663bfff9db54293591412a298adb1cbc65141bca7f7c916d5ca489c8f6118dd8`.
- `cell_02_block1_on/cell_result.json`: `a6ada9632ae7f0777bdfbddb500ca4ac715a8b2ff1ffc7d14f99f091b847ee1f`.
- `cell_03_block2_on/cell_result.json`: `bbd9eab691e4d13622fc89a5e6c2e31a223b04de5b94fc269fd87d7da1f88b5f`.
- `cell_04_block2_off/cell_result.json`: `8775180416812662aa4aa4c67939baa27a7eb8789e9f94cda29929bd9db4f007`.
- `cell_05_block3_off/cell_result.json`: `5cd3b9d04db26a1adba7a38fe32d065348ecf4caf61fc4981386ff5a96353d07`.
- `cell_06_block3_on/cell_result.json`: `b7815faf73876e01b90e2214c08892cbb756465ddae1515cbeffb9b37ac3eb53`.

The independent JSON preserves per-prefix coverage, raw API streams, per-request clipping/termination, ownership summaries, head counts, exact scalar analysis and all terminal hashes. Prior reports retain each raw join and source hash. For final cells05/06, ledger hashes are respectively `bd5f7042271b9fa3ffe9efe6772410393f6159e7ee67c4d1e0624636b54b811e` and `71a91c6bcaa0041795523b6410d4fcbe515c3c1c89f0d493b9daee65e0091a8e`; join hashes are `8c233712f809530af63d7a4e70aa229e36d2c83e6d8b189d49c024fa40544065` and `1187026c4d87dfd292a05ad05e75b86cf461f858825ccaa9e5fc03e38d412e10`.
