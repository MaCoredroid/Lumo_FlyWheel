# Focus Candidates

## schema-snapshot-alignment
- Fix the mismatch between snapshot writer ordering and restore ordering.
- Still the blocker for adaptive batching and trustworthy future replay.

## contained-pager-cleanup
- Looks loud because a stale screenshot from before containment is still circulating.
- Fresh pages are already at zero after the rate limiter and fallback guard.

## cold-start-cache-tuning
- Helpful p95 work, but not the thing blocking the next roadmap milestone.

## streaming-watermark-alignment
- Useful follow-on work, but still downstream of stable snapshot restore.
