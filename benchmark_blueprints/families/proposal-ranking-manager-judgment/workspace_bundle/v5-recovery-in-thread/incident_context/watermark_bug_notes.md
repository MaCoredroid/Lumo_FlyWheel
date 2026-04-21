# Streaming watermark — current state

- The streaming watermark module `src/respproxy/watermark.py` is owned
  by the realtime-ingest team, not respproxy.
- A redesign of the watermark is in realtime-ingest's Q3 plan. Expected
  merge: week 10 of Q3 at the earliest.
- Until the redesign lands, any respproxy work that reaches into
  watermark via cross-thread coordination risks the same class of
  deadlock as `INC-2481`.

## Implication for proposal ranking this cycle

- **P2 (L2 response cache)** — cache invalidation crosses the
  watermark boundary. Cannot re-attempt until the redesign lands,
  unless it is rearchitected to not depend on the watermark.
- **P4 (schema-cache warm-start)** — does not touch the watermark.
  Safe to proceed.
- **P5 (validator microservice)** — does not touch the watermark
  directly; the validator call sits before streaming starts.
  Safe to proceed.
- **P1 (drop structural validation)** — does not touch the watermark.
- **P3 (Rust rewrite)** — would re-implement the watermark integration
  from scratch; depends on the watermark redesign schedule regardless.

The incident does not change the staffing picture. P3 is still blocked
on the Rust specialist; P4 is still the most surgical path.

## Streaming reliability win

A proposal that *also* reduces streaming-reliability incident rate is
more valuable this cycle than a pure latency-win proposal. By that
lens, P5 has become a higher-value pick than P4 even though P4 has
better latency numbers: P5's service boundary isolates validation
from the watermark coupling and removes a future coupling surface.

## Ranking preview (author's note for the manager — not binding)

If this note were a recommendation rather than evidence: P5 first
(reliability isolation + reasonable latency), P4 second (surgical
latency win, no watermark exposure), P2 demoted (watermark-coupled),
P1 and P3 unchanged.
