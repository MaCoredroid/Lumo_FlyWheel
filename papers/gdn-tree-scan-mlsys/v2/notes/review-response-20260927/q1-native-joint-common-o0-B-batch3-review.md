# Native joint common-O0 B: fixed indices 84–131 raw review

**PASS for exactly 48 observations, frozen indices `[84,132)`.** All 48 independent raw audits pass and equal their saved driver audits. Together with the two preserved, disjoint batches, this establishes 132 reviewed B observations. Full 168-observation B completion, A/B reduction, candidate qualification and any timing claim remain pending.

Run: `q1-native-joint-common-o0-20260929T155000Z-aligned_nonpacked-B`. At **2026-09-29 20:47:22 UTC**, all first 132 frozen observation IDs had a seal and completed, sealed driver record. Selection then used exactly indices 84–131 in case/R2 order, with no outcome filtering or replacement. Errors would have been retained at their original indices. The audit finished **20:54:44 UTC**, after **442.55 seconds** of audit work; command exit 0, empty stderr and zero errors.

Coverage is **24 cases × R2**: 28 observations at the medium prefix (28,941 tokens), followed by 20 at the long prefix (60,083 tokens). The unchanged accepted raw auditor is SHA256 `9f28dd94d2fc243c70e42769573b5c33e5613ce749c95113efc0e72ff9aae992`. Its checks cover target O0/O1 state and logical KV, target O2 full logits, common source/import/bootstrap/continuation joins, MTP raw history and same-input native checks, consumed input/position/owner bindings, and the declared categorical phases. Additional driver checks verify observation/case/repeat identity, prompt parity/counts, per-observation salt, successful seal authentication, absent HTTP errors and exact equality with the saved audit. Every selected seal and driver file retained its SHA during its audit.

All **46 campaign source paths plus two external script pins**, and **19 immutable metadata files from the accepted B boot review**, matched before and after the batch. The complete 84-case/R2/168-request B job was independently reproduced. The batch script reuses batch 2's checks with only the fixed selection and completeness/count constants changed; the invocation explicitly pins CPUs 18–19.

| Artifact | SHA256 |
|---|---|
| Corpus | `ac3f1248aeb5343714334b402c9f0cd6ecaca856c68cae8c625a5c9b5b7b8f61` |
| Joint-source manifest | `49ec5c4ddb97342cf2ca4db41be2027d1044366e9fa3cb08b0017961a38795b8` |
| B job | `de44fb103d7184ea3046c78887b736ecb749aac59fe218554fdd3996bb88b666` |
| Saved B gate | `03f4a0fb515f32688fef6433661dd09a0daf00832d342d1e3bff678aeeffd794` |
| Exact batch-3 audit script | `0e8ddf9d8d731bc1b5e7e0bbc8bd316edb459add7ecaaa3b08d53eabe4b69b0e` |
| Independent JSONL output | `68213d39920e4080383aca3ec99ff98f6ace4c791b6ec54e5c9bcecf1adfa0ee` |

The shared target/bootstrap object reader authenticated **92,566 distinct objects / 16,056,377,344 bytes** within this batch. MTP/source per-row counts overlap; they are not summed with this count, across batches, or represented as archive size.

Local evidence preserves every observation's categorical projections, ordered top-3/spine values, ties/margins and input/position/owner bindings. The separate reference ledger retains **192 phase-vector references**, representing **96 distinct full 248,320-element vectors / 95,354,880 bytes**. The stored representation is little-endian float32, with native dtype retained. The unchanged raw auditor authenticated those bytes remotely. No new full-vector local mirror was requested or created; authoritative remote raw objects remain retained.

At audit end, **138 seals and 138 driver files** were visible, with **no terminal run receipt**. This inventory does not admit any observations beyond the selected 48. A local metadata check verifies that batches 1–3 contain 132 distinct observation IDs, exactly the first 132 frozen IDs in order. It does not rerun prior raw audits or constitute a full A/B reduction. No extra batch was started.

The task used read-only remote CPU work on affinity **18–19**, OMP/OpenBLAS/MKL thread limits **2**, empty CUDA visibility and disabled bytecode writes. No remote files, sources or gates were changed; no inference request or model/GPU/container/cache action was performed. Existing boot findings were reused through immutable metadata hashes; no new live Docker/GPU status query was issued. Evidence is retained under `p0/monitor/review-response-20260927/native-joint-common-o0-B-independent/batch3/`, including command receipts, complete JSONL, source/boot pins, selection records, summary, vector references and a hash seal. Prior evidence remains unchanged.
