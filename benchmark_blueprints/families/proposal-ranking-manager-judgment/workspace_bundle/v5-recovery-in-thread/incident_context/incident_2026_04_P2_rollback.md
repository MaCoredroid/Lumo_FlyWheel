# INC-2481 — P2 rollback

**Severity:** SEV-2 (degraded experience across multiple skill packs)
**Started:** 2026-04-07 18:14 UTC
**Rollback initiated:** 2026-04-09 02:30 UTC
**Rollback complete:** 2026-04-09 04:10 UTC
**Owner:** respproxy team

## Summary

The L2 response cache (P2) went into staged rollout on 2026-04-04.
Within 72 hours, 0.9% of streamed tool-call responses showed truncation
or stall behavior identical in symptom to `INC-2411 / 2419 / 2427`,
but in new skill packs that had been stable. The common factor was
cache invalidation triggering a cross-thread read of the streaming
watermark.

## Root cause

The L2 cache invalidation path acquires locks in order
`(invalidation_lock, watermark_read_lock)`.

A streaming-reliability fix that landed 2026-03-25 (two weeks before
P2 rollout) introduced the reverse order on the watermark path:
`(watermark_read_lock, watermark_publish_lock)`.

When a skill-pack version bump triggered invalidation during an in-flight
stream, the two code paths deadlocked on watermark_read_lock briefly;
the deadlock detector broke one, the stream surfaced a truncation to
the client, then retried. Hence the 0.9% stall-or-truncate rate.

## Why it was not caught in shadow replay

Shadow replay did not include the streaming-reliability fix landed on
2026-03-25; the L2 cache shadow-replay numbers in
`repo_evidence/perf/l2_cache_shadow_replay.md` pre-date that change.

## Learnings

- The invalidation-to-watermark coupling, flagged as the "primary
  risk" in the prior ranking, was the actual cause. The risk note
  identified the surface but not the specific lock-ordering failure.
- Any proposal that depends on strong invalidation semantics against
  the current watermark module must be paused until the watermark
  redesign lands — or must be rearchitected to avoid the watermark
  coupling entirely.
- Proposals that isolate validation or caching behind a service
  boundary are less exposed to this coupling.
