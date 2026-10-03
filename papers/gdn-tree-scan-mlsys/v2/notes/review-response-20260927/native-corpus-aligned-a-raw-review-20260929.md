# Native corpus aligned A: independent complete raw review

**One-run validation passed: 168/168 observations, zero raw or provenance failures.** This covers `q1-native-corpus-calibration-20260929T045821Z-aligned_nonpacked-A` only. It does not qualify the primary native baseline: the frozen four-run A/B × aligned/packed reduction remains pending, and packed remains a separate control.

## Complete raw and provenance checks

The independent CPU audit ran on the completed remote archive from10:24:36.758293 to10:25:05.008667UTC (28.250s), with OMP/OpenBLAS/MKL thread ceilings of2. No model request, GPU action, container action, cache recovery or scientific-source mutation was performed.

It authenticated frozen `q1_native_corpus_v1.py` SHA `d4502b7f47ab1ac6ed8cd295aa5565726fad486c4a1ff1bda6a13fe24e4ea770` and executed its unchanged per-run reducer loop, including its full corpus/source/design comparison. The small AST adapter omits only the four-distinct-run precondition and final cross-run aggregate. It preserves the original per-run loop AST exactly and returns rows under an explicit single-run label; the omitted AST statements and loop hash are retained in `AST-SCOPE.json`.

All168 expected observations were found, with no extra/missing/invalid case file. The audit verified record seals, job/control/request/case/repeat and prefix/chain identities, actual consumed tokens and positions, pending-token boundary, every O0/O1 layer and group, expected geometries/dtypes, within-request storage continuity, and complete logical KV coverage. It authenticated and finite-checked **58,579 unique content objects totaling19,923,152,896 bytes**, exactly matching the terminal object-store count and size. It recomputed all168 complete248,320-entry logit vectors' smallest-ID argmax, top values, ties and margin from raw bytes and matched the recorded decisions.

All **374 terminal metadata members** matched their declared sizes and hashes. The unchanged loop verified the approved one-run gate, frozen launch/source identities, actual immutable image, actual engine environment/command/mounts/resource limits against the pinned renderer, no engine restart, native non-speculative aligned route, 48 GDN +16 attention layers, patched FA2 installation and dispatch, and native Runner patch anchors/content. Every driver request had authenticated seal and exact sent/usage prompt-token parity.

The terminal receipt reports `COMPLETED_driver_rc=0_cleanup_rc=0`, engine stopped and removed, driver exit0, and completed168/168. Actual retained container inspection shows exited, exit0, not OOM-killed and restart count0. Container ID is `55cb3eb5804c787134a26749cdb12ec5364052d0423910373f0279c7a35b92d8`; image is `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. Terminal receipt SHA is `f774b75b348f22a03bc1a98e56b3d37891ebbe28398b6ce1bf9244a2cee507dd`.

Retained readiness admission also binds the same run, gate and frozen readiness source to the existing SUCCESS/exit0 operational receipt, all true boot conditions and successful finalization. It is historical operational evidence, not a new resource observation or recovery action. Its receipt SHA is `62be09ed00e1e3d1f262f1ee2f5adc19d7748a574c6cdf559114bd73b0ae7f41`; the supplemental `RETAINED-METADATA.json` records the producer's actual `outcome` field and source/gate checks.

## R2 results, grouped by the same case

| Check across r0/r1 | Result |
| --- | --- |
| Complete case pairs | 84/84 |
| Same greedy decision | 84/84 |
| Byte-identical O0 logical state | 84/84 |
| Byte-identical O1 logical state | 84/84 |
| Byte-identical complete logit vector | 84/84 |
| Exact-tie observations | 2/168 |

Both exact ties belong to `calibration-long_available__c0__n05` (r0 and r1): two maximal logits, margin0, and smallest-ID winner279 in both. These are not missing/invalid observations; they remain in the fixed denominator and must remain visible in later comparisons. No tie was silently dropped.

The driver's `repeat_o0_digests_identical=false` and `repeat_o2_argmax_identical=false` are **global summaries over all168 receipts**, as the frozen `q1_reference_driver_v2.py:130–131` shows. There are three distinct O0 logical states and nine distinct winners across different cases. Those fields therefore do not establish per-case instability. The independently grouped R2 table above is the applicable within-run characterization; cross-process A/B behavior is still unobserved here.

## Evidence and limits

Artifact directory: `p0/monitor/review-response-20260927/native-aligned-a-independent-20260929/`. It contains the executable adapter, frozen reducer copy, supplementary read-only metadata checker and `audit-20260929T102500Z/` outputs. The output includes all168 row summaries, all84 per-case pairs, exact raw counts, source hashes, AST scope and retained metadata. Local copies were authenticated against the remote audit manifest. Original audit manifest SHA is `85e147d3ce61d325fe7c58a5d09b891e7151428f5aaa973314ca446f21c39742`; a separate review manifest binds the supplement and this note without rewriting that original manifest.

Corpus SHA is `f0320a57bdec8fb9aaf878313972ea391f936fe48a8e4ad46ddacb3375e40e44`; job SHA is `757b995f39b2616c7f60412c3a0ec974810ad171b55304261683bb6640329652`; fixtures SHA is `607634e19235a58d2b6d73f8260dac3c49f842f9f1cba9a1789f329c56012223`.

This result supplies one fully validated aligned-A calibration run. It makes no candidate, MTP, lifecycle, held-out, continuous-cycle, timing or agent-workload claim and changes no scientific gate or fixed denominator.
