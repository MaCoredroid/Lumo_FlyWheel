# Round Plan

## Selected Focus: Schema Snapshot Alignment (`schema-snapshot-alignment`)

The abandoned cache-tuning patch is tempting, but it does not remove the replay invariant that keeps future work from sticking. The next round should fix the prerequisite before touching the partial perf path.

### Focus Citations
- `focus_candidates/priority_map.md`
- `objective_metrics/scoreboard.md`
- `repo_state/open_threads.md`

## Do Not Touch
- `contained-pager-cleanup`: Still contained and not the gating dependency.
- `cold-start-cache-tuning`: The in-progress patch is abandoned partial work, not a head start; finishing it before the invariant is fixed would optimize a path that still replays incorrectly.

## Partial-Progress Metric
- Metric id: `snapshot-drift-repros`
- Baseline: Replay diverges in 7 of 10 seeded restore runs and the cache patch still sits half-wired behind a local flag.
- Target: Replay diverges in 0 of 10 runs while the abandoned cache patch remains deferred and untouched this round.
- Guardrail: Do not merge or extend the cold-start cache patch until replay determinism is proven stable.
- Measurement plan:
  - Run the seeded restore replay suite after every ordering change.
  - Verify the cache patch diff hash stays unchanged while replay divergence drops to zero.

## Assumption Ledger
- `observed` The partial cache patch does not unlock the blocked launch items: Repo-state explicitly says the downstream work is still blocked by replay determinism.
- `to_verify` Cache tuning may be the follow-on round once snapshot replay is stable: The focus map still lists it as worthwhile, just not first.
- `missing` Whether the abandoned patch contains any safe-to-reuse benchmark harness pieces: The bundle shows the patch but not which pieces could be cleanly salvaged later.
