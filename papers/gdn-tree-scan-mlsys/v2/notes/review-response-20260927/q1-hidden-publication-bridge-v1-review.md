# Hidden-publication bridge v1 — bounded independent review

Disposition: PASS for the inspected source-only bridge after incremental partial-capture retention repair. No remaining concrete blocker found in actual-operand joining, stage ordering, padding, normalized-return/gather identities or retained failure payloads. This helper is not yet wired into Eagle; it neither proves live producer origin nor grants numerical/launch qualification. Parent owns implementation and live admission.

Reviewed bridge SHA256: `00d30bc67e34027b09eae5a0386017ffc7f8766d0a7399218abb5920a15622b8`.

## Checked behavior

- Runner target must be the same-storage32-row prefix of the retained parent, then the exact same identity/bytes as the source of the buffer copy. The copy oracle checks source preservation and logical buffer-tail preservation.
- The actual hidden kwargs must be a same-storage prefix of that copied buffer. Both32-row and40-row padded inputs pass; an equal-valued clone at foreign storage refuses. Padding bytes are tied to the captured destination, so the copy cannot pass using an unused buffer.
- Current methodmtp requires hidden kwargs and non-tuple returns. Actual return, normalized sampling source and normalized continuation source must be the same object. Both gather sources are joined back to that return, so unrelated self-consistent selected rows cannot stand in for the actual result. The helper correctly permits the two sources to alias.
- Stage/owner lists require the unique ordered runner→copy→forward→sampling→continuation path. Changed owner/indices, wrong route, source mutation, missing continuation and early/failed forward evidence refuse. Final audit recomputes raw equations rather than trusting saved success flags.

## Partial-retention finding and closure

Initial source review found multiple captures assembled in one expression before assignment to the retained record. A later capture exception could hide an earlier successfully stored capture's metadata; constructor failure could occur before `self.record` existed. Parent repaired the implementation before the first independent fixture run copied the source.

Current constructor publishes a skeleton through `record_sink` before owner lookup/capture; each successful capture is assigned immediately. Copy/forward/gather subrecords exist before fallible capture. The gather source reference remains retained until both after-captures succeed. Injected controls verify:

1. Second constructor capture failure retains the first parent entry through the same sink record.
2. Copy-before second capture failure retains its source entry/subrecord.
3. Forward-after second capture failure retains the actual returned entry.
4. Sink failure propagates before owner lookup or any capture.
5. Selected-output capture failure retains source-after bytes and its live source reference; no successful audit is set.

The first fixture script still expected pre-repair behavior and failed that stale assertion against already repaired bytes. Its script and explanation are preserved; it is not presented as a reproduced pre-repair false pass.

## Independent evidence and boundaries

`p0/monitor/review-response-20260927/hidden-bridge-independent-20260929/` retains source snapshots, scripts, raw JSON results and a manifest. Ten injected stdlib controls pass: two complete positive bridges and eight expected refusals, plus five explicit failure-retention controls. Fake byte tensors use the real bridge and existing raw-byte oracle; no Torch, model, first forward, GPU or live request was executed by this review.

Parent's separately retained Torch attempt1 log reports nine methods passing in0.118s. Its test source executes the pinned real copy/gather AST statements with a synthetic CPU arithmetic forward. Parent identified that run as preceding the retention repair; this review does not use it as current-source verification or claim it was independently rerun.

## Required live-collector obligations

These are the helper's existing external contracts, not new experiment criteria:

- Supply `owner(stage)` from actual Runner/proposal/pending/payload registry observations, with the stage-aware consume/clear handling in `q1-hidden-publication-live-wiring-review.md`. A caller-supplied owner dictionary alone is not evidence.
- Supply `record_sink` that retains the same mutable record on the active case before construction can fail. Persist the current record and exception on failure; retaining only an initial serialized/deep-copied skeleton loses subsequent capture updates.
- Mark the active process/case unusable on every bridge/capture/sink error; no retry or later completion may erase that failure. This helper does not independently own the collector's fail-stop lifecycle. Its stage list records callback entry, not proof that an incomplete stage succeeded.
- At offline case audit, derive expected owner/path from separately bound case/event receipts, not solely from the bridge's own fields. Bind helper/collector/patcher/Eagle source hashes and actual producer/consumer callbacks before claiming live publication.

Original operations, accepted hidden-byte criteria, cohort, model and serving settings remain unchanged.
