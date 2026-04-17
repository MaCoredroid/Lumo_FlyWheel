# Oracle — project-board variant

This directory contains the authored oracle used to prove that the broken
`project-board` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` repairs the visible cross-layer task by threading owner data
  through the store, service, CLI, and docs while adding `owner_source` plus a
  whitespace-stable `routing_key`.
- `solution_followup.patch` tightens canonical routing so punctuation-heavy
  project-board owner labels normalize into a stable routing-key prefix without
  mutating the visible `owner` field.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The project-board-specific trap is that a straightforward visible fix often
builds `routing_key` from the raw owner text. That keeps the visible
`pm-oncall` fixtures green, but it leaves separator-heavy owner labels unstable
for downstream board routes.
