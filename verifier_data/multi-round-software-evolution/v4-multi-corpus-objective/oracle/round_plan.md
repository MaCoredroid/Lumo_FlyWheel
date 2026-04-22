# Round Plan

## Selected Focus: Streaming Watermark Alignment (`streaming-watermark-alignment`)

The release gate shifts the primary objective: the next launch is blocked more by streaming watermark mismatches than by raw replay drift. This round should move the gating customer objective first, while still respecting the invariant risk noted in the bundle.

### Focus Citations
- `focus_candidates/priority_map.md`
- `objective_metrics/scoreboard.md`
- `release_context/release_gate.md`

## Do Not Touch
- `contained-pager-cleanup`: Still contained and not release-blocking.
- `cold-start-cache-tuning`: The partial cache patch remains a side quest relative to the release gate.

## Partial-Progress Metric
- Metric id: `watermark-mismatch-repros`
- Baseline: Streaming watermark mismatch reproduces in 6 of 20 release-gate replay runs and blocks the customer launch checklist.
- Target: Streaming watermark mismatch reproduces in 0 of 20 release-gate replay runs while snapshot drift does not regress above its current 7 of 10 baseline.
- Guardrail: Do not treat the abandoned cache patch as progress and do not let snapshot replay drift get worse while chasing watermark parity.
- Measurement plan:
  - Run the release-gate replay pack and count watermark mismatches after every candidate fix.
  - Track the base snapshot-drift replay counter in parallel so the objective shift does not hide invariant regressions.

## Assumption Ledger
- `observed` The release gate makes watermark parity the immediate objective: Release context names it as the blocker for the next customer rollout.
- `to_verify` Snapshot drift may still be a prerequisite risk for the watermark fix: The release context elevates watermark parity, but repo-state still warns about the invariant under it.
- `missing` Whether the launch can tolerate partial improvement instead of zero mismatches: The bundle states the blocker, but not the exact acceptability threshold short of zero.
