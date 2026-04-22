# Round Plan

## Selected Focus: Schema Snapshot Alignment (`schema-snapshot-alignment`)

Round 4 replay still diverges because the snapshot writer and restore path disagree on field ordering. Fixing that invariant unblocks adaptive batching and keeps future streaming work from being built on sand.

### Focus Citations
- `focus_candidates/priority_map.md`
- `objective_metrics/scoreboard.md`
- `repo_state/open_threads.md`

## Do Not Touch
- `contained-pager-cleanup`: The pager storm is already rate-limited and no longer blocks the roadmap; reopening it this round would burn time without unlocking the blocked follow-on work.

## Partial-Progress Metric
- Metric id: `snapshot-drift-repros`
- Baseline: Replay diverges in 7 of 10 seeded restore runs after round 3.
- Target: Replay diverges in 0 of 10 seeded restore runs and adaptive-batching shadow runs stay byte-stable.
- Guardrail: Do not regress watermark mismatch rate above 1 in 20 replay runs while fixing snapshot ordering.
- Measurement plan:
  - Run the seeded restore replay suite on every snapshot writer change.
  - Compare the byte diff summary and watermark mismatch counter before declaring the round complete.

## Assumption Ledger
- `observed` Adaptive batching launch stays blocked until snapshot restore is stable: Repo-state note ties the launch checklist directly to stable snapshot replay.
- `to_verify` Cold-start cache tuning remains secondary once snapshot drift is fixed: The perf win matters, but the current scoreboard says it is not the gating dependency.
- `missing` Whether watermark parity still needs extra fixtures after snapshot alignment: The current bundle does not say if the same snapshot fix is sufficient for all watermark edge cases.
