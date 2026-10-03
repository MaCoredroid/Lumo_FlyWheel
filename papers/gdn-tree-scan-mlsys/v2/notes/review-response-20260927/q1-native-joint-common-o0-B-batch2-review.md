# Native joint common-O0 B: fixed indices 20–83 raw review

**PASS for exactly 64 observations, frozen indices `[20,84)`.** All 64 independently recomputed joint raw audits pass and equal their saved driver audits. This is a partial process-B review only: no full A/B admission, complete native eligibility, candidate qualification, timing result or denominator change.

Run: `q1-native-joint-common-o0-20260929T155000Z-aligned_nonpacked-B`. At **2026-09-29 20:09:27 UTC**, every one of the first 84 frozen observation IDs had both a seal file and a completed, sealed driver record. The audit then selected exactly indices 20–83 in case/R2 order. It did not filter or replace observations based on outcome, and would have retained each row error at its original index. It finished **20:14:20 UTC** after **293.30 seconds** of audit work; command exit was 0 and stderr was empty. No errors occurred.

Coverage is **32 cases × R2**: 36 observations at the short prefix (13,487 tokens), then 28 at the medium prefix (28,941 tokens). Raw validation reused the accepted unchanged auditor SHA256 `9f28dd94d2fc243c70e42769573b5c33e5613ce749c95113efc0e72ff9aae992`. It covers target O0/O1 state and logical KV, target O2 full logits, authenticated common source/import/bootstrap/continuation joins, MTP raw history and same-input native checks, exact consumed input/position/owner bindings, and the declared categorical phases. Driver checks also establish matching observation/case/repeat identities, prompt-token counts, per-observation salt, successful seal authentication and absence of HTTP errors. Every selected seal and driver file retained its SHA during its audit.

All **46 campaign source paths plus two external script pins**, and **19 immutable metadata files from the accepted B boot review**, matched before and after this batch. The complete 84-case/R2/168-request B job was reproduced; this partial review does not replace that run scope. Bound identities:

| Artifact | SHA256 |
|---|---|
| Corpus | `ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61` |
| Joint-source manifest | `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8` |
| B job | `de44fb103d7184ea3046c78887b736ecb749aac59fe218554fdd3996bb88b666` |
| Saved B gate | `03f4a0fb515f32688fef6433661dd09a0daf00832d342d1e3bff678aeeffd794` |
| Exact batch-2 audit script | `b90b8207358a1e49963bd5a7cfefdcd77f23025653af182c1d82432ca6ca6297` |
| Full independent JSONL output | `1cf3fba69a85d6c29d5334f68e99dd9d9c1cfbb4ecb737114b787d23daabdaca` |

The shared target/bootstrap object reader authenticated **47,006 distinct objects / 11,212,562,432 bytes**. MTP and accepted-source checks additionally validate their raw objects, but their per-row counts overlap and are not added to this unique total or treated as archive size.

The local JSONL retains every selected observation's categorical projection, actual spine/ordered-top3 values, ties/margins and phase input/position/owner bindings. A separate reference ledger preserves all **256 phase-vector references**, representing **128 distinct complete 248,320-element vectors / 127,139,840 bytes**. The recorded payload representation is little-endian float32 with native dtype retained. These payloads were authenticated remotely by the unchanged raw auditor; no new local vector mirror was requested or created. The authoritative remote content-addressed objects remain retained.

At audit end, **106 seal files and 106 driver files** were visible, and **no terminal run receipt** existed. Those counts are an instantaneous inventory, not a review of observations outside this fixed batch. No extra batch was started. The existing B boot review was reused through immutable metadata hashes; no new live Docker/GPU status query was performed.

The remote task ran read-only on CPU affinity **18–19**, with OMP/OpenBLAS/MKL thread limits **2**, empty CUDA visibility and bytecode writes disabled. No remote files, sources or gates were changed; no inference request or model/GPU/container/cache action was performed. Evidence is local under `p0/monitor/review-response-20260927/native-joint-common-o0-B-independent/batch2/`: exact script and delta from batch 1, command start/finish receipts, full JSONL, reused boot/source identities, summary and categorical vector references. Prior batch and boot evidence remain unchanged.
