# Candidate sequence runtime — initial bounded review

**Required repairs on original runtime `1fb7585ad4b7395ebd1ec072cc042472efcca6ea59c8b883f9104097409c1b11`.** This report preserves the original findings; the parent is implementing a successor in the same prospective file. No launch or qualification approval.

## Concrete findings

1. **Frozen prefix/source binding omitted at case start.** `begin_sequence_case` compares control, supplied prefix, actual prompt and independently authenticated source against one another plus the fixed record length, but does not reconstruct the record's frozen prefix source and padding. A coherent foreign same-length prefix is not excluded by this function. The parent independently identified this gap. Also latch the canonical authenticated source/root/prefix metadata and recheck it immediately before the first import, so in-memory replacement cannot substitute another source after validation. The source validator itself is unchanged/previously accepted.
2. **Retained O2 copy can diverge from its actual sealed forward.** Powered through the full retained CPU callback flow: before terminal seal change `_sequence_pending_o2[0].argmax_smallest_id` from 7 to 999. Completion still succeeds; the corresponding actual next-forward sealed `target_root` remains 7. Require exact O2 population, private digest preservation and equality to the subsequent sealed/current forward head.
3. **`zip` silently truncates receipt authentication.** Before the final seal, change a prior `target_o1.materialized_tokens` and clear `_sequence_record_digests`. The loop authenticates zero pairs, and completion returns four records with only one digest. Require exact cardinality and expected cycle order before comparison, including at final completion. The control returns extent13500 for a planned13499 without marking the process unusable.

The latter two are recorded in `RECEIPT-RESULTS.json`, alongside a passing unmodified flow. The fixture uses real accepted forcing/publication/graph/event callbacks but explicitly isolates model state/KV snapshots, matching the parent's provided test scope. These are receipt integrity failures, not numerical-model findings.

## Independently powered pre-forward/import coverage

Eight additional controls execute actual Maps, Joint/H4 import, full 48-layer target-state snapshots, 16-layer target KV reblocking and MTP KV snapshot logic using real CPU tensors. Their schedule is deliberately rebased to the retained65-token synthetic source fixture; it is not submitted to the frozen production input loader and cannot become a scientific input population.

- Initial joint import occurs exactly once, and full target/MTP logical digests match the source. Production int32 prepared IDs remain unchanged through read-only int64 cursor projection.
- Carrying into a subsequent prepared forward retains one import total and two live-map snapshots. No interior hydration occurs in this control.
- Duplicate initial forward, corrupt source SSM bytes, a changed materialized table prefix and wrong position axis all refuse. Corrupt source bytes fail before the mutation latch.
- A post-import readback mismatch and a post-import `KeyboardInterrupt` both raise `ProcessUnusable` in the same callback with the mutation latch set, preserving the need to discard that process even though import itself completed.

Evidence: `PREFORWARD-EXTENDED-RESULTS.json` and `preforward_controls_extended.py`. No GPU/model, remote process, cache operation or gate was used. Original source/tests are preserved under `sequence-runtime-review`. Local Torch is Python3.9, so the CPU harness supplies semantically equivalent strict-zip and positive-integer bit_count behavior only in preserved fixture namespaces; initial Python-version setup failures are retained in logs. The reviewed runtime source is unmodified.

The corrected Graph MTP position source is a separately reviewed dependency; this review does not re-decide its primary model semantics. No singleton, launcher, serialized raw admission, full-model qualification or workload admission follows from these checks.
