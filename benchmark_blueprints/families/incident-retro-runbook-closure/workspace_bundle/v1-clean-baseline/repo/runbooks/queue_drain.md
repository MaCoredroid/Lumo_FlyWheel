# Queue Drain Closure

Use this runbook after the queue depth returns to zero.

## Required closure sequence

1. `queue-drain quiesce --cluster atlas-a`
2. `queue-drain audit-post-drain --cluster atlas-a --include-stuck-shards`
3. `python3 repo/scripts/queue_drain_helper.py --emit-shard-report atlas-a`
4. `Escalate to #queue-drain-war-room with the shard report if any shard remains stuck`

## Escalation

Escalate unresolved stuck shards to `#queue-drain-war-room` with the shard report.
