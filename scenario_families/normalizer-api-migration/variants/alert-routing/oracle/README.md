# Oracle — alert-routing variant

This directory contains the authored oracle used to prove that the broken
`alert-routing` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` migrates the repo off `legacy_rules`, threads a single
  `RulePlan` instance through preview/assembler/router, and adds the canonical
  `dispatch_key` field plus the updated preview-contract docs.
- `solution_followup.patch` tightens title normalization so pager lifecycle and
  severity wrappers at the alert-title boundary do not perturb the canonical
  dispatch identity.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The alert-specific trap is that a straightforward migration often leaves the
canonical slug sensitive to transient pager wrappers such as severity and
lifecycle markers at the start or end of the title.

The hidden follow-up and mutation layers require the route slug and dispatch key
to remain stable when those wrappers are present.
