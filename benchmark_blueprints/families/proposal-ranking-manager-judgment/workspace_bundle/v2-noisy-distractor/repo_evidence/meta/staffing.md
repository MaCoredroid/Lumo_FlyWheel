# Staffing plan — respproxy team, Q3 2026

**Closed:** 2026-04-10 (Q2 planning cycle)
**Approved by:** Mei (EM), signed off by VP Engineering
**Status:** plan of record for Q3 2026 capacity on respproxy

## Current headcount on respproxy

| Engineer | Role                 | Q3 2026 allocation                                  |
| -------- | -------------------- | --------------------------------------------------- |
| Priya    | Senior backend       | **100%** — tech lead on respproxy                    |
| Ravi     | Backend              | **100%**                                             |
| Mei      | Platform infra       | **100%** — lead on platform                          |
| Anja     | Backend              | Part-time until 2026-05-08, then 100%                |
| Kenji    | Platform             | **Departing to transformer-runtime end of 2026-05** |

## Cross-team loans

| Engineer | Home team            | Availability for respproxy      |
| -------- | -------------------- | ------------------------------- |
| Diego    | transformer-runtime  | **0%** — 100% booked on the    |
|          |                      | quantization release through Q3 |
|          |                      | and into Q4                     |

## Rust experience

- Diego: primary Rust specialist. Unavailable (see above).
- Nobody on respproxy team ships Rust as a first language.
- Hiring backfill for a Rust specialist is not in the Q3 plan.

## Footnote

Any proposal whose lead is departing the team during the rollout window
requires an explicit handoff plan recorded in the proposal itself. A
proposal without a handoff plan is not staffable as-written.

## Implication

Any proposal that requires a Rust specialist, or a full Rust rewrite, is
not staffable in Q3 without pulling Diego off the quantization release.
Leadership has signaled that the quantization release is not pauseable.
