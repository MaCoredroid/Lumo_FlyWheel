# Staffing update — respproxy team, Q3 2026 mid-quarter revision

**Issued:** 2026-06-15
**Issued by:** Priya (tech lead) + Mei (EM)
**Supersedes:** the per-engineer availability table in `staffing.md`. The
headcount list and the Rust-specialist note are unchanged.

## What changed since the Q2 planning snapshot

`staffing.md` was closed on 2026-04-10 and captured the state of the team
at that point. Two availability changes have landed since:

| Engineer | Previous availability     | Revised Q3 availability        |
| -------- | ------------------------- | ------------------------------ |
| Priya    | Available (lead)          | **40% Q3** — parental leave    |
|          |                           | transition; on-ramp + handoff  |
|          |                           | begins 2026-07-08              |
| Anja     | Part-time until 2026-05-08 | Fully available from 2026-05-12 |

Ravi and Mei are unchanged. Diego remains **0%** on respproxy through Q3
and into Q4 (quantization release is still the sole priority for the
transformer-runtime loan).

## Implication for any proposal owned by Priya

Any proposal authored by Priya — including P4 (schema-compile warm-start) —
is now constrained by Priya's 40% Q3 availability. That is still enough
to keep P4 on track if the plan accounts for it:

- Ship behind a staged rollout so Priya is not the single point of
  review during ramp windows.
- Ravi can cover day-to-day implementation during Priya's ramp-down
  starting 2026-07-08; this is pre-approved.
- Any plan that assumes Priya full-time or as the only reviewer of a
  rollout gate must be re-planned.

## Implication for any proposal owned by another engineer

P1 (Ravi), P2 (Mei), P3 (Diego), and the validator microservice work
(Kenji → TBD) are unaffected by Priya's change. Staffing constraints
noted in `staffing.md` for those engineers still apply.

## How to reflect this in a manager brief

If your accepted proposal is owned by Priya, the brief should:

1. Cite this update file among the accepted entry's `citations`.
2. Acknowledge reduced Q3 availability in `primary_risk` or in the
   accepted rationale (keywords: `40%`, `parental leave`, `reduced
   availability`, `mid-quarter`, `handoff`, `Ravi cover`).
3. Name the staffing risk in `assumption_ledger` or in mitigations
   (e.g. `Ravi cover during Priya ramp-down`, `staged rollout so Priya
   is not the sole reviewer`).

If your accepted proposal is owned by someone else, this update only
matters insofar as it confirms capacity is NOT freeing up on Priya's
side — no action required beyond being aware.
