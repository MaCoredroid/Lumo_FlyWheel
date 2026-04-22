# Focus Candidates

## schema-snapshot-alignment
- Fix the mismatch between snapshot writer ordering and restore ordering.
- Unlocks adaptive batching and future streaming work because replay determinism becomes trustworthy again.
- Quiet blocker: customers do not see it directly, but every follow-on round keeps tripping over it.

## contained-pager-cleanup
- Visible cleanup on the pager storm path from round 3.
- Already contained by the rate limiter and fallback guard.
- Good cleanup, wrong round if it is already contained.

## cold-start-cache-tuning
- Improve cold-start cache hydration latency.
- Helpful perf work, but not the current gating dependency.

## streaming-watermark-alignment
- Align resume watermark handling across restore and live streaming.
- Useful follow-on work once the underlying snapshot restore is trustworthy.
