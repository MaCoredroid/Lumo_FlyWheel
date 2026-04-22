# Round Plan

## Selected Focus: Schema Snapshot Alignment (`schema-snapshot-alignment`)

The stale pager snapshot makes the contained issue look loud again, but the current scoreboard and open-thread note still show snapshot drift as the blocker for any durable next step.

### Focus Citations
- `focus_candidates/priority_map.md`
- `objective_metrics/scoreboard.md`
- `repo_state/open_threads.md`

## Do Not Touch
- `contained-pager-cleanup`: The noisy alert count is from the pre-rate-limit snapshot. Current pages are contained, so this should stay deferred unless fresh paging escapes the guardrail.

## Partial-Progress Metric
- Metric id: `snapshot-drift-repros`
- Baseline: Replay diverges in 7 of 10 seeded restore runs while fresh pager pages remain at 0 in the last 7 days.
- Target: Replay diverges in 0 of 10 runs with fresh pager pages still at 0 and no new restore-order diffs.
- Guardrail: Do not reopen the contained pager path or reclassify stale snapshots as current signal.
- Measurement plan:
  - Track seeded replay divergence and the fresh pager counter from the current scoreboard together.
  - Require both replay stability and zero fresh-page regressions before closing the round.

## Assumption Ledger
- `observed` The stale pager snapshot is not a current blocker: The scoreboard marks it historical and the round log says the rate limiter contained it.
- `to_verify` The schema ordering fix may reduce some watermark noise as a side-effect: There is a plausible overlap, but the current evidence does not quantify it.
- `missing` Whether any fresh customer report bypasses the pager guardrail: No post-containment customer report is included in the bundle.
