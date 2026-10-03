# Repaired common-O0 A: complete independent raw audit

**PASS for all 168 observations in this one process.** All 84 cases have complete r0/r1 pairs with identical imported common O0, resulting O1, full-vocabulary logits, and smallest-ID greedy decisions. This validates process A's retained evidence only. Process B and the frozen cross-process reduction remain required; full Q1, MTP, candidate, lifecycle, held-out and workload qualification remain false. The earlier natural-prefill A/B adverse result and failed common-O0 v1 attempt are unchanged.

Run: `q1-native-common-o0-repair2-20260929T125100Z-aligned_nonpacked-A`. The independent audit started only after `RUN-RECEIPT.json` appeared with `COMPLETED_driver_rc=0_cleanup_rc=0`. It ran 13:48:26.256414–13:49:26.733460 UTC (60.477 seconds), CPU only, with OMP/OpenBLAS/MKL ceilings of 2. No model, engine, Docker, GPU, cache, source or scientific-gate action was performed by this reviewer.

| Check | Authenticated result |
| --- | --- |
| Expected / driver / raw-valid observations | 168 / 168 / 168 |
| Complete r0/r1 case pairs | 84 / 84 |
| Common O0 / O1 / full-logit equality within pairs | 84 / 84 for each |
| Stable smallest-ID greedy decision within pairs | 84 / 84 |
| Raw current-run objects | 107,396; 27,127,390,208 bytes |
| Fixed source O0 provenance | all 84 original A/r0 records; 47,743 objects; 6,676,615,168 bytes |
| Terminal metadata members | 374, all size/hash verified |
| Frozen source payloads | all 27 match corpus/freeze/launch |
| Exact-tie observations | 2, both repetitions of long/n05 |

The tie case is `calibration-long_available__c0__n05`: winner 279, tie count 2, margin 0 in both repetitions. Its full logits and O0/O1 are identical across r0/r1. No tie-breaking policy or tolerance changed.

The full existing common-O0 v2 auditor authenticated each canonical seal, owner/control/prefix/chain, natural bootstrap, imported O0, O1, full raw O2 and exact decision; checked every referenced object's hash, length and dtype-specific finiteness; and enforced native groups/rows/maps/storage, active attention allocation, alias topology and import-receipt bindings. The independent audit separately authenticated all 84 selected original source record files and canonical seals, then recomputed every source O0 logical digest from raw content. No case was removed, replaced or reclassified to obtain this result.

Natural bootstrap and imported common state remain distinct: 81/84 natural-bootstrap pairs are identical, while the short, medium and long root-only cases have different natural-bootstrap digests between repetitions. **All 84 imported O0 pairs nonetheless match their exact frozen source O0 and produce identical O1/logits within A.** This is why natural-prefill state must not be confused with the shared imported boundary. No cause of the bootstrap difference is inferred here.

The driver's two global `repeat_*_identical=false` fields do not indicate a per-case failure: source lines 144–145 compare one set across all 168 receipts, mixing different prefixes and paths. The independent results group by the exact case ID and require both r0/r1 identities; all 84 pairs pass. The raw driver fields are retained unchanged.

## Boot, configuration and cleanup

The frozen per-run loop verified the full corpus reconstruction, actual job builder output, gate/launch/source hashes, actual immutable image and container identity, rendered engine configuration against retained Docker argv/environment/mounts/resources, stock-runner patch reproduction, FA2 installation/dispatch, driver receipts, and terminal completeness. The retained native boot reports spec-off aligned/unpacked B1, 48 GDN and 16 FA2 attention layers, FP32 SSM, and kernel block sizes `[1024,1024,1024,64]`.

Actual retained image: `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. The created, stopped and removed CID is consistently `ece7c243f6ee7855dd9fe6eeb83add7d2bebf7aa46b1719ecdd6641d6f828809`. Final inspect has `Running=false`, `ExitCode=0`, `OOMKilled=false`, `RestartCount=0`, and empty error. The exact-CID cleanup state is `stopped_and_removed`, with empty cleanup stderr. The retained post-run GPU-process CSV contains only its header; this is a historical observation, not a new live query.

Receipt timestamps: engine started 12:55:54, healthy 13:01:06, driver ended 13:47:29, owned cleanup stopped 13:47:33 UTC. Boot was 312 seconds; driver collection was 2,783 seconds (46m23s); engine start to cleanup was 3,099 seconds (51m39s). These are untimed qualification operational intervals, not serving speed measurements or workload throughput. The readiness receipt also authenticates to the exact run, gate and source, with successful finalization.

## Audit boundary and retained evidence

`audit_full_run.py` extracts the frozen corpus reducer's per-run loop verbatim and asserts AST equality. Only the two-run cardinality requirement and final across-process aggregation/return are omitted, with their exact source and lines recorded in `AST-SCOPE.json`; every per-run success, raw and terminal predicate remains. This produces an explicitly single-run result and cannot award cross-process qualification. The original reducer and runtime sources were not edited.

Artifacts are in `p0/monitor/review-response-20260927/native-common-o0-repair2-a-independent-20260929/`: `full-attempt1/RESULT.json`, `ROWS.json`, source-O0 provenance, AST scope, execution log, final retained metadata, and final review seal. Only small summaries, selected early JSON records, source snapshots and terminal metadata were copied locally; the 27 GB raw object store remains on the source host. The four-observation early review remains separately sealed.

Key SHA-256 bindings:

- Terminal: `eccb6d60d0779df76226c52d61e66bbe8259fe38178349a4ff74c6fd0c6c5b31`.
- Source freeze: `ceb210445886cd334c5b266a171d16a8877ad77dde599cd28f451413608371cd`.
- Corpus: `28d79abdd4d1b824bb778d70e3b33b993bc11d1a936cd9d27a0e5b99c9701682`.
- Frozen reducer: `bc4e04b0b6cabcc0a4f10ab63e9fd3859ce2721a22dac0a9505bf441362c1c37`.
- Independent script: `baa3079c462b18c2216bca86cc0a81a75a61ecc9b76686c0e5ef39e0da5b1bc4`.
- Independent result: `2537e907d061e67ff89b13199bb4751ca4b07e57d4205af16ccda2b1a4b62e6d`.
- Full audit manifest: `ddd3d8bee9c566aec0096700382312d8643f6d7faa8bc7543630cc73e95121d3`.
- Final retained metadata: `df46d9b7951996183ff99ab06f248d689d38c933dcbd28b269c6116853cc41d7`.
