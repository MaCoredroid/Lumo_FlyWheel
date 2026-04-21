# L2 cache shadow-replay — 2026-03-28 through 2026-04-11

Methodology: shadow traffic tee from prod respproxy into a test pod that
answers from the proposed L2 cache. No live impact.

## Hit-rate by skill pack (aggregate across the window)

| skill pack        | qps  | cacheable qps | hit rate |
| ----------------- | ---- | ------------- | -------- |
| incident-handoff  | 180  |  160          | 71%      |
| writing-desk      |  95  |   72          | 58%      |
| code-reviewer     | 220  |  140          | 44%      |
| retrieval-lite    | 140  |   95          | 39%      |
| tool-sandbox      |  55  |   11          | 12%      |
| **aggregate**     | 690  |  478          | **53%**  |

## p95 latency projection

At observed 53% aggregate hit rate: p95 drops from 420ms to ~240ms
(hits skip compile + validate + dispatch; misses pay same cost as today).

Warm-window projections (hit rate > 65%) would bring p95 below 200ms.

## Invalidation touchpoints

The proposed invalidation semantics touch:
 - skill-pack version bump (clean; one hook per release)
 - tool-registry update (clean; one hook per change)
 - per-entry TTL 90s
 - streaming watermark module (for safety on stream-to-cache transitions)

The watermark module is owned by the realtime-ingest team. A team-owned
redesign of the watermark is pending in their Q3 plan. Coupling here is
the primary rollout risk.

## Risks

 - Cross-version leak if invalidation races the rollout.
 - Increased memory cost; initial budget 2GB/pod, monitored.
