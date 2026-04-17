# Oracle — release-gate variant

This directory contains the authored oracle used to prove that the broken
`release-gate` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` repairs the visible cross-layer task by threading owner data
  through the store, service, CLI, and docs while adding `owner_source` plus a
  whitespace-stable `routing_key`.
- `solution_followup.patch` tightens canonical routing so punctuation-heavy
  release train labels normalize into a stable routing-key suffix without
  mutating the visible `name` field.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The release-gate-specific trap is that a straightforward visible fix often
normalizes only whitespace in the release train name. That keeps the visible
`Patch Train` fixtures green, but it leaves versioned or separator-heavy gate
labels unstable for downstream routing.
