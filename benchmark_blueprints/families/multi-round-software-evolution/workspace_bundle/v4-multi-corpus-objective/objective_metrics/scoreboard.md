# Current Objective Metrics

- Replay divergence after seeded restore: 7 / 10 runs
- Fresh pager pages last 7 days: 0
- Cold-start cache p95: 263 ms
- Streaming watermark mismatch in release-gate replay pack: 6 / 20 runs

The release gate elevates watermark mismatch to the primary visible blocker.
