# Oracle — catalog-sync variant

This directory contains the authored oracle used to prove that the broken
`catalog-sync` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` migrates the repo off `legacy_rules`, threads a single
  `RulePlan` instance through preview/assembler/router, and adds the canonical
  `dispatch_key` field plus the updated preview-contract docs.
- `solution_followup.patch` tightens title normalization so sync-source labels
  already encoded by owner and region metadata do not perturb the canonical
  dispatch identity.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The catalog-specific trap is that a straightforward migration often leaves the
canonical slug sensitive to source labels echoed by upstream sync jobs at the
start or end of the title.

The hidden follow-up and mutation layers require the route slug and dispatch
key to ignore those redundant source wrappers while preserving product issue
tokens and digits.
