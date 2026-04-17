# Oracle — warehouse-queue variant

This directory contains the authored oracle used to prove that the broken
`warehouse-queue` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` repairs the visible cross-layer task by threading owner data
  through the store, service, CLI, and docs while adding `owner_source` plus a
  whitespace-stable `routing_key`.
- `solution_followup.patch` tightens canonical queue routing so separator-heavy
  warehouse queue names normalize into a stable routing-key suffix without
  mutating the visible `name` field.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The warehouse-queue-specific trap is that a straightforward visible fix often
normalizes only whitespace in queue names. That keeps the visible `picker
backlog` fixtures green, but it leaves separator-heavy labels such as
`Picker / Backlog` and `Picker::Backlog` unstable for downstream queue routes.
