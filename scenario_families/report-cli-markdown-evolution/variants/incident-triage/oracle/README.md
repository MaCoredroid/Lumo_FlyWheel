# Oracle — incident-triage variant

This directory contains the authored oracle used to prove that the broken
`incident-triage` repo is solvable and that the hidden verifier bundle stays
aligned with the intended behavior. Never mount these files into an agent
container.

## Files

- `solution.patch` adds the Markdown renderer, wires it through the CLI, and
  updates the usage docs.
- `solution_followup.patch` fixes the latent ack-SLA boundary trap in the
  shared incident model so threshold-equal incidents count as breached.

## Why two patches?

`incident-triage` deliberately differs from the other report variants.
Round 1 is the authored feature uplift: JSON already works, the shared
summary layer already exists, and the missing work is the Markdown handoff
path.

The latent bug is not about zero counts or dormant watchlists. The incident
runbook in `docs/triage-policy.md` says the ack SLA is inclusive at the
threshold, but the shipped `Incident.is_ack_sla_breached()` helper only marks
an incident breached when it is strictly greater than the threshold. Hidden
round-1 fixtures avoid threshold-equal incidents, so a good round-1 solution
still leaves that bug intact.

The follow-up patch is the real fix site because both JSON and Markdown
consume the shared summary layer. Renderer-local workarounds still fail the
hidden follow-up tests.

## Expected states

| Repo state | visible tests | hidden bundle |
| --- | --- | --- |
| broken repo | fail | fail |
| `solution.patch` applied | pass | round-1 slices pass, follow-up slices fail |
| `solution_followup.patch` applied on top | pass | full bundle passes |
