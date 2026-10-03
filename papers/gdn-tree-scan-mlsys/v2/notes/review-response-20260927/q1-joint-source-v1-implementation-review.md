# Joint native source v1: bounded independent source review

**PASS for the new joint-source capture/structural-join/driver wiring.** No concrete blocker found in this delta. This establishes source/CPU readiness for capture only; no joint import exists in this scope, no source inventory or launch is approved, and no native or candidate numerical qualification follows.

## Ordering and source identity

`q1_reference_hooks_joint_source_v1.py:10–18` extends the reviewed v2.4 MTP hook. The inherited v2.4 `on_pre_forward` calls its base target observation first, which dynamically calls the new `_snapshot_state`; only after that returns does v2.4 call `History.pre` for the root. Thus the new capture sees the completed previous prefix first pass while `History.step`, `pending`, and `Program.live` are still clear. It does not accidentally capture after arming/executing the root MTP pass. Independent AST execution of both actual class methods in one inheritance chain confirms target snapshot → current MTP snapshot → root history preparation.

`q1_joint_source_v1.py:14–33` requires the live history/owner/generation, exact P extent, last first-pass record with shifted next token equal to r, current request/observation, matching last lease, no pending work, correct target O0 tag/extent, and fixture root. It rereads the singleton MTP K/V through the reviewed full-prefix exporter and requires the entire current snapshot dictionary to match the prior prefix record, including physical/logical mapping and raw content references. The return binds the target digest, MTP digest, prefix/root boundary, prefix record, and complete prefix record list. Subsequent appended history is not included in that frozen prefix hash.

The target snapshot precedes the current MTP snapshot but neither performs a model update; both are read-only captures with the existing stream readback behavior. An observed MTP prefix mismatch invokes `_mtp_fail` and raises; the inherited path cannot proceed to root `History.pre`. Independent connected control verifies owner/history failure, ready=false, invalid sealing, and no root preparation. The unchanged loader/cache owner and native history were not requalified here.

## Offline join and driver

`audit_join` (35–53) binds the same sealed observation/request, source P/r, target O0, quiescence labels, selected prefix record and all prefix-history hashes, exact current/recorded MTP snapshot, group/extent and joint digest. The future or follow-up history cannot be substituted as the prefix. The caller remains responsible for authenticating the enclosing seal/job/fixture and raw target/MTP payloads; this helper is deliberately structural, not a standalone raw auditor or proof that execution occurred.

The new driver differs from accepted v3 by its schema and a mandatory `audit_join` invocation immediately after the existing full MTP raw audit. Existing target object authentication precedes both. Independent execution of the actual `authenticate_seal` with injected raw-audit outcomes confirms: valid path calls raw audit then join; a raw-history failure stops before join; missing joint source and changed MTP boundary data refuse. Injection was only for call-order/failure controls and supplied no raw numerical evidence. The surrounding source-corpus builder must still authenticate/derive complete target logical digests and freeze the actual source selection before import, as already required by the design.

Metadata tampering controls cover foreign request/generation/boundary/root position, pending work, missing join, source-root or lease mismatch, altered prefix K/V or hidden lineage, and rehashed wrong shifted input. These tests do not claim resistance to an attacker fabricating every mutually consistent record; hash seals provide integrity/provenance when bound to the actual reviewed executable and run artifacts.

## Patcher and tests

The patcher was applied in memory to exact pinned native worker SHA `904d7d357ba076c93cefee5f7f030e2be577dc12599aebdf56395b894e2077b0`. The generated Python compiles, retains all five reference and seven owner anchors, and has SHA `205799b54ec8bbae7641f8f5e47f88d78d43adbe8b4a512c0866bc604274db10`. Replacing only the new hook import with the accepted v2.4 hook import gives the accepted worker-v3 patch output byte-for-byte. Already-patched and wrong-source input refuse. No emitted worker was executed.

The 11 supplied standard-library CPU methods passed locally. Sixteen additional independent controls passed; their complete names and patch receipt are in `INDEPENDENT-CONTROLS.json`. Tests use synthetic metadata and explicit base/raw-audit facades where named, with actual new functions, actual v2.4 pre-forward control flow and actual patcher construction. No Torch/GPU/model/container/cache operation, implementation edit, run-source mutation, gate action, or experiment-count increment occurred.

## Exact reviewed bytes

- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/q1_joint_source_v1.py`: `2fb11fd09b50c421fc192146790ca5aa0edbf78e4b57ed0b73c48a292f1b151e`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/q1_reference_hooks_joint_source_v1.py`: `22edc9f985203a9a1f4c76744297ebfbfc475f06b0b678133a3264076c798414`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/q1_patch_native_mtp_joint_source_v1.py`: `cc37c6cae13512f736af8444f0948db9a8920ecf36c0c8213b7561201e85f427`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/q1_reference_driver_joint_source_v1.py`: `3ea559dd482a09862be15afcf611b3853adf8dca83ce48abd3821e30179a9c8f`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/tests/test_q1_joint_source_v1.py`: `9118ae49cdb6c57c275f9e24f26af37aa7d9f8d026a1d48698a4665d67b3457b`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/independent_controls.py`: `4c8a727d4ad4db38ea653daf9ea1b56861ab5a35e5d4741abf9a25e5e3dc2b53`
- `p0/monitor/review-response-20260927/joint-source-v1-implementation-review/INDEPENDENT-CONTROLS.json`: `00ddedd028b8857e36c0b09d3013ff0317a65e558978e1893b4ae8baa3641098`
