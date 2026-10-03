# Joint common-O0 primitive and native mapper v1 review

**One required entry-point closure before calling this an end-to-end single-use import.** The union import, raw authentication, combined digest, target/MTP separation, and in-transaction interruption handling passed bounded CPU tests. No numerical criteria, model execution, import history integration, or launch approval is reviewed here.

## F1 — mapper failures occur before the single-use transaction starts

`q1_native_joint_common_o0_v1.py:118–128` invokes `live_destinations` and checks source extent before `Transaction.apply` is called. The transaction marks `started=True` only at `q1_joint_common_o0_import_v1.py:47`. Consequently an early mapper refusal does not mark the transaction started or failed, and it can be retried.

The independent actual-Torch control uses the valid native-shaped fixture at P=65, changes `runner.seq_lens` from 66 to 67, and calls the public native import entry point. It raises RuntimeError as expected, but the transaction remains `{started:false, failed:false, mutation_attempted:false}`. Restoring sequence length66 then successfully imports through the same transaction. This failure occurs after `owner.assert_live_cache` has returned, so an actual owner's fail-stop decorator does not close it; it is not a synthetic-owner-only artifact. No bytes were written during the refused attempt.

Minimal correction: let the transaction own the complete mapper/preflight/apply attempt, or explicitly consume/fail it around the native mapper and extent checks. Every exception, including BaseException, must leave the attempt unavailable for reuse. Retain authentication-before-write and the distinct mutation-attempt flag. The future hook must still mark its process unusable and stop the forward on any failure; a primitive's internal flags alone cannot stop an engine. Do not work around this by catching the mapper error and creating another transaction inside the same observation.

## Accepted bounded behavior

`combined_source` checks exact target48/16 coverage, joint source boundary/quiescence/extent, original target aggregate and per-layer digests, MTP singleton digest, and the distinct joint digest before constructing a temporary 17-attention import view. An independent control confirms the original source object/target digest is not mutated and the combined digest is different. Full manifest selection, source-run authentication and prefix/root admission remain explicitly caller-owned; this primitive is not a replacement for the source sealer/auditor.

The implementation calls unchanged H4 once with GDN48 and attention17, keeps `expect_alias=(16,3)`, and supplies the MTP singleton to the same authorized-write union, complement guards, all-object authentication and final all-layer readback. The native owner/mapper admits only target48/16 plus the exact singleton; current group/registry/cache/block units and owner/proposer/generation enter the identity snapshot. An explicit conservative byte-span check refuses MTP/target overlap before H4 copies. No target-only intermediate success is returned.

The existing guard coverage remains bounded. It includes the added MTP prefix-tail/neighbor guards and excludes more distant pages/scratch according to the receipt. It does not become a whole-cache check. Native H4 cross-family target alias handling was reused, not requalified or weakened. The combined union uses the target's real alias classes and a disjoint MTP cache.

Eight supplied actual-Torch CPU methods passed locally under miniforge Python/Torch2.8.0 (1.748 seconds). They cover exact joint import/readback and one latch, MTP corruption refusing before target write, overlap/source-digest/registry/block-unit refusal, guard corruption, identity replacement, and no reuse after entered transactions.

Seven additional independent controls passed as expected, including the F1 witness. Authentication interrupted with KeyboardInterrupt leaves all checked destination bytes untouched, failed=true and no mutation latch, and refuses reuse. Interruption from the mutation callback marks failed and mutation_attempted before any write. Interruption after copying leaves the observed MTP bytes changed, failed=true/completed=false, and refuses reuse. These BaseExceptions propagate in their original type; callers must catch/finalize BaseException and consult the latched state, not assume every failure is an H4 ImportFailure. A changed MTP allocation generation after latching also refuses. The tests do not simulate arbitrary thread races or process death, and no such stronger guarantee is claimed.

All fixtures were private temporary CPU data. No GPU/network/container/model operation, implementation edit, shared source-object change, gate update, or experiment-count increment occurred. The initial reviewer F1 probe used a foreign-layer change; it was replaced before the final recorded run by the sequence-length probe to avoid relying on a stubbed owner's registry semantics. Both used the same frozen implementation. The historical source snapshot and logs are retained separately from any successor repair.

## Evidence hashes

- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/q1_joint_common_o0_import_v1.py`: `7f836ed52a8a3bc1acb88b419f7b583ec905072d7ed3928cd63af3e549ba4cf6`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/q1_native_joint_common_o0_v1.py`: `f080be8da13839861b87ecde9459c32e7b3fbcedaaa1671dd824636348870c79`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/tests/test_q1_joint_common_o0_import_v1.py`: `5e532f6c329c61d510003065f3758172fc022f539a37da369929ea0563cb29af`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/independent_controls.py`: `668c1a0b3e5c3a0793d88924b109e52e24139c3f3b2641d024d4ade50351d3f6`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/INDEPENDENT-CONTROLS.json`: `f6d614b53b5f1f6552045ef4405ea7f29df248e7f0f39a089564458e13a573b0`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/supplied-test-log.txt`: `3c2242fcdf8bfa14226fe1094946bc5c759b696a7db5da10c766d5d1a014f058`
- `p0/monitor/review-response-20260927/joint-common-o0-import-v1-review/independent-test-log.txt`: `f6d614b53b5f1f6552045ef4405ea7f29df248e7f0f39a089564458e13a573b0`
- `experiments/review-response-20260927/tools/q1_candidate_hydration_v4.py`: `b35f2e5d49e0f49827103db3b0c3ddecdec6ce38189883a7c4a4c13c79350fac`
- `experiments/review-response-20260927/tools/q1_native_common_o0_v2.py`: `a080f7623b83fc2f3a144bbc86b1d233d1f0d58a76bcd3b4f31e46cf0a6f3f63`
- `experiments/review-response-20260927/tools/q1_native_mtp_owner_v2.py`: `77edbafbd9ff34813d1a866c5ab8c8c7c620da0c3ce1e532d4b41fa1e61cde7e`
