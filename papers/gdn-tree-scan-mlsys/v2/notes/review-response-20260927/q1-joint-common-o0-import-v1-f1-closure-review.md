# Joint common-O0 import F1 closure

**PASS for F1 and the bounded primitive/native-mapper source review.** Final native mapper SHA `38d3e9f95c6fcc1143578380fcccfc022706a990ccc01d324f81e6d6ab9ab78b`; unchanged joint primitive SHA `7f836ed52a8a3bc1acb88b419f7b583ec905072d7ed3928cd63af3e549ba4cf6`. The historical F1 finding/reproducer remains in the separate original review.

The only implementation delta wraps the complete native mapping, source-extent admission and existing apply call in a BaseException handler that marks the attempt started and failed. An upfront `Import.need` rejects already consumed transactions with the existing H4 ImportFailure subtype. No row/geometry/cache-map/source-digest/numerical rule, H4 copy, guard or readback code changed. A transient repair used the native RuntimeError guard instead; the final source fixes that refusal subtype and the positive/reuse control verifies it.

Four independent local Torch2.8 CPU controls pass: (1) seq_len67 at P65 refuses and consumes the attempt; repairing the metadata to66 cannot reuse it; (2) pre-apply source extent mismatch consumes the attempt; (3) KeyboardInterrupt in mapping consumes/fails the attempt without mutation; (4) the existing full union import succeeds with one mutation latch and subsequent use raises ImportFailure. These are source/CPU checks with private native-shaped fixtures, not GPU execution. The prior eight supplied and seven independent controls for unchanged union/digest/authentication/complement/identity/interrupt behavior remain applicable; they were not unnecessarily rerun wholesale.

The future hook remains responsible for a process-level unusable latch and stopping forward execution on all errors, including BaseException. This closure establishes single-use behavior for the reviewed public native entry point; it does not implement or approve history import, source-manifest admission, a live caller, or a launch. Source admission and actual source-root/fixture compatibility remain the caller's declared obligations. No implementation, gate, runtime or experiment counter was changed by the reviewer.

## Evidence hashes

- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/q1_native_joint_common_o0_v1.py`: `38d3e9f95c6fcc1143578380fcccfc022706a990ccc01d324f81e6d6ab9ab78b`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/q1_joint_common_o0_import_v1.py`: `7f836ed52a8a3bc1acb88b419f7b583ec905072d7ed3928cd63af3e549ba4cf6`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/tests/test_q1_joint_common_o0_import_v1.py`: `77b6d2cbf1a60c3d403c5996dab6f6c39caccd29e70f13f247d16c1066f8c399`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/closure_controls.py`: `2d24ed2c7531b6fa9fcff7f57f9b4f5344a6477e54dc4957fb61d2e64ca7d959`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/CLOSURE-CONTROLS.json`: `d12341a5ce9807207f8e403db05f220e9f6a7c8cfbb2b51c79b8ff0667c32b33`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/f1-repair/closure-test-log.txt`: `d12341a5ce9807207f8e403db05f220e9f6a7c8cfbb2b51c79b8ff0667c32b33`
