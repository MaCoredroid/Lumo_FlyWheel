# Native common-O0 v1 implementation review

Disposition: **one required repair before source readiness**. The fixed native row adapter, outcome-independent source selection, and transactional hook are sound within the inspected scope, but the prepared sequence/query metadata premise is not enforced. This is a prospective source/CPU review; it authorizes no launch, changes no criterion, and supplies no GPU or numerical qualification evidence.

## F1 — authenticate the real one-token metadata, not only its selected row

`q1_native_common_o0_v1.py:65–145` verifies align mode, the 48/16 layer registry, four cache groups, native cache geometry, and equality between the metadata state row, request block ID, CPU table, and GPU table. However, it never reads the actual GPU `runner.seq_lens`, the actual query interval, or the GDN decode/prefill counts. `q1_reference_hooks_common_o0_v1.py:33–46` checks scheduler count one and all prepared position axes, but reads only the query start (not the end or GPU counterpart).

The independent control executes the actual functions with tensor-metadata facades. The valid B1 case has P=65, GPU sequence length 66, query interval [0,1], and correct native rows 1/2/3. All these malformed cases are nevertheless accepted: GPU sequence length 67; absent sequence length; GDN query interval [0,2]; GDN num_decodes=0; and, at the hook, stale GPU sequence length or CPU/GPU query interval [0,2]. This is seven recorded false admissions across the two functions. A stale sequence length within one 1024-token column can retain the correct selected state row, so the four-way equality alone does not prove the real root forward consumes the intended continuation boundary.

This requirement is already in the accepted row-derivation design. Pinned `gpu_model_runner.py:1986–1992` prepares positions and GPU sequence lengths from computed plus scheduled tokens; the GDN builder `gdn_attn.py:170–175` consumes that sequence length, and `:199–210` produces decode counts, selected state indices, and the non-spec query interval. The previous note `q1-native-gdn-state-row-derivation-review.md` explicitly requires GPU seq_len=P+1.

Minimal repair: before mutation, require the actual B1 CPU/GPU query interval to be [0,1], actual GPU sequence length to be P+1, and actual per-layer GDN metadata to describe one decode token, zero prefill/speculative tokens, and the same query interval. Bind these prepared metadata values/identities into the before/after identity recheck so they cannot silently change during import. Keep the existing scheduler, actual-token, all-axis-position, and four-way row checks. Add powered missing/stale/wrong-count controls; do not change H4 or any numerical rule. At subsequent forced single-token steps, the hook should retain the corresponding computed+1 sequence/query checks.

## Accepted bounded behavior

- Wrong metadata row, request row, CPU/GPU table, Mamba column, and all-mode refuse in independent controls. Historical null GDN columns remain permitted; only the selected column must be positive.
- The adapter reuses the unchanged H4 global plan, preauthentication, alias census, exact complements, all-layer readback/digest and identity callback. It does not reuse candidate FR13 row ownership.
- Actual subclass controls with a narrow Base facade demonstrate successful import clears the in-progress latch, while pre-write failure, post-write failure, and an independent readback mismatch all seal invalid, latch unusable, raise, and refuse the next forward. These are CPU control-flow checks, not execution of real H4 copying.
- `source_for_case` authenticated all 84 actual archived A/r0 records against the newly generated source manifest and their original file/seal/job/prefix/continuation identities. Selection-rule, missing/duplicate population, file hash, record seal, logical digest, job identity, and traversal mutations all refused. This pass read record JSON only; it did not reread the 6.68 GB raw source payloads or rerun the parent's source-builder raw audit.
- The builder binds the complete original 84-case population without filtering by outcome and labels target-only/no-MTP/no-launch. The historical natural-prefill 0/84 result remains explicit. The parent's supplied full native-shape Torch suite reports five passing methods; that log was inspected, not independently rerun on this host (Torch unavailable).

Independent artifact `INDEPENDENT-CONTROLS.json` contains 29 controls, including the seven deliberately expected false admissions establishing F1. The facade uses full layer naming and geometry metadata but no allocated GPU/model tensors. An initial reviewer-script invocation used the wrong ancestor to locate SOURCES.json and stopped after the function controls; the path was corrected before the successful recorded run. No implementation, gate, source corpus, or runtime was changed.

## Exact review bytes

- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/q1_native_common_o0_v1.py`: `9a7af54eb2c5c5ea8d18bdf1976f3ba8cb87c2ab4d9964165e32994af73daa5d`
- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/q1_reference_hooks_common_o0_v1.py`: `9eaedce27d00ac6206bb4ee2f1a3e4efa172828457bb5fcb174ec3dc0ea1fb21`
- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/q1_native_common_o0_sources_v1.py`: `8437b602e0d25facd3cab036af849adb3699cae0f3974fb49314c4987762b945`
- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/tests/test_q1_native_common_o0_v1.py`: `6061d86d1aa779bedd3134936f8f4459fea0d3992963eb80cdbb33be36a28e44`
- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/independent_controls.py`: `967c9b76ae61a90dffc3e6557a7c51509245567387feab75e9ce5d07fb03f2d3`
- `p0/monitor/review-response-20260927/native-common-o0-v1-implementation-review/INDEPENDENT-CONTROLS.json`: `f3b09de2ccb445d51e312b4f960884e18aa8c045ced320155886bdde2640e032`
- `experiments/review-response-20260927/fullmodel/native-common-o0-v1/SOURCES.json`: `e4f7e3bb7944f197a62b02bad6fa85ca3b4f1792db9be3471b5e02855dc7e343`
- `experiments/review-response-20260927/tools/test_log.native_common_o0_v1.attempt1.txt`: `486e0e76df171f83e5a573519f557277f78368f58a81a0c1e1176253621a8151`
