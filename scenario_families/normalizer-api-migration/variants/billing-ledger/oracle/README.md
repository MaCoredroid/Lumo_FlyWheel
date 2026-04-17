# Oracle — billing-ledger variant

This directory contains the authored oracle used to prove that the broken
`billing-ledger` repo is solvable and that the richer hidden verifier bundle
stays aligned with the intended two-step solve path.

## Oracle layout

- `solution.patch` migrates the repo off `legacy_rules`, threads a single
  `RulePlan` instance through preview/assembler/router, and adds the canonical
  `dispatch_key` field.
- `solution_followup.patch` tightens title normalization for separator-heavy
  billing exports so the slug and dispatch key use single-hyphen canonical
  tokens.

## Discrimination story

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | follow-up + mutation slice still fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |

The billing-specific trap is that a straightforward migration often normalizes
titles with simple whitespace collapsing. That is enough for the visible suite,
but it still produces non-canonical slugs for ledger-export titles such as
`Refund / Retry _ Queue` and `Chargeback---Retry!!!`.

The hidden follow-up and mutation layers require the route slug and dispatch key
to collapse separator noise into single hyphen-delimited tokens while
preserving digits.
