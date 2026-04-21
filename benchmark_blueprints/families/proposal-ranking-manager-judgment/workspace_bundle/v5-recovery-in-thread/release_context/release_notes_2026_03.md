# respproxy — release notes, 2026-03 cycle

## What landed

- FP8 quantization rollout on the transformer-runtime. Model_forward on
  tool-call path dropped from 95ms to 42ms. (See `meta/rollout_history.md`.)
- Minor: respproxy v0.19.0 tagged on 2026-03-15. No behavior changes.

## What was downgraded

- **Streaming-reliability backlog** (category: `streaming`) — downgraded
  after the incident wave in 2026-02 stabilized. Three tickets rolled to
  Q3 under "watch and revisit."

## What reopened during the cycle

- **2026-03-28:** `INC-2411` — customers reporting stalled or truncated
  streamed tool-call responses. Frequency: ~0.4% of streaming tool calls.
  Severity: degraded-experience, not outage-grade. Owner: respproxy.
- **2026-03-30:** `INC-2419` — similar, different skill pack. Suspected
  same root cause.
- **2026-04-02:** `INC-2427` — identical symptoms. Three within two weeks.

## Implication for Q3 planning

The streaming-reliability downgrade is no longer tenable. The Q3 plan
recorded the latency objective as top-priority because it was set in
early March, before the incident wave reopened. Re-prioritization is on
the agenda for the next q-planning checkpoint.

The priority order for Q3 is therefore:

1. **Streaming reliability** on the tool-call path (newly urgent).
2. Latency reduction on the tool-call path (still important but not
   blocking; user-facing p95 at 420ms is not a reported complaint).
3. Backlog cleanup.
