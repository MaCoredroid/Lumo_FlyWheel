# Roster memo — respproxy team

**Issued:** 2026-06-15
**Issued by:** Priya (tech lead) + Mei (EM)
**Supersedes:** the per-engineer Q3 2026 allocation table in `staffing.md`.
The headcount list and the Rust-specialist note there are unchanged.

## What changed since the Q2 planning snapshot

`staffing.md` was closed on 2026-04-10 as the Q3 2026 plan of record and
captured expected allocations at that point. Two availability changes
have landed since:

| Engineer | Plan-of-record allocation | Revised Q3 availability       |
| -------- | ------------------------- | ----------------------------- |
| Priya    | 100% (tech lead)          | **40% Q3** — parental leave   |
|          |                           | transition; on-ramp + handoff |
|          |                           | begins 2026-07-08             |
| Anja     | Part-time through         | Fully available from          |
|          | 2026-05-08                | 2026-05-12                    |

Ravi and Mei are unchanged. Diego remains 0% on respproxy through Q3
and into Q4 (quantization release is still the sole priority for the
transformer-runtime loan).

## Implication for proposals authored by Priya

Any proposal authored by Priya is now constrained by Priya's reduced
Q3 capacity. The reduction is workable if the plan accounts for it:
staged rollouts that do not require Priya as a single-point reviewer,
and Ravi covering day-to-day implementation during Priya's ramp-down
starting 2026-07-08 (this cover is pre-approved at the EM level).

Plans that assume Priya full-time — or that put Priya on the critical
path as the only reviewer of a rollout gate — need to be re-planned
before they can ship in Q3.

## Implication for proposals owned by others

Proposals owned by engineers other than Priya are unaffected by this
change. Capacity is NOT freeing up on Priya's side; staffing constraints
noted in `staffing.md` for those other engineers still apply as written.
