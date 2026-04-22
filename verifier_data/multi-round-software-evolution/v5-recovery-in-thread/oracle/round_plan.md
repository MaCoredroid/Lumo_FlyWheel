# Round Plan

## Selected Focus: Schema Snapshot Alignment (`schema-snapshot-alignment`)

The watermark plan was already tried and rolled back because the unstable snapshot invariant underneath it corrupted recovery state. The next round should step back to the prerequisite invariant fix before retrying watermark work.

### Focus Citations
- `focus_candidates/priority_map.md`
- `repo_state/open_threads.md`
- `incident_context/rollback.md`

## Do Not Touch
- `contained-pager-cleanup`: Still contained and still not the thing that failed the last attempt.
- `streaming-watermark-alignment`: Do not retry the rolled-back watermark plan until snapshot restore ordering is stable; otherwise the team repeats the same incident path.

## Partial-Progress Metric
- Metric id: `snapshot-drift-repros`
- Baseline: The watermark round rolled back and replay still diverges in 7 of 10 seeded restore runs.
- Target: Replay diverges in 0 of 10 runs and the rollback reproduction no longer corrupts recovery state before any watermark retry is scheduled.
- Guardrail: Do not reopen the rolled-back watermark plan until the rollback reproduction passes cleanly and fresh customer streaming regressions stay flat.
- Measurement plan:
  - Re-run the rollback reproduction after every snapshot-ordering change.
  - Require both zero restore divergence and a clean rollback replay before re-queuing watermark alignment.

## Assumption Ledger
- `observed` The rollback was caused by retrying watermark work on top of unstable snapshot state: Incident context names snapshot ordering as the prerequisite that was skipped.
- `to_verify` Watermark alignment may become the immediate next round once snapshot restore is stable: That depends on whether the rollback reproduction stays clean after the invariant fix.
- `missing` Exact exit criteria for safely retrying the rolled-back watermark plan: The incident writeup explains why the retry failed, but not the formal retry gate.
