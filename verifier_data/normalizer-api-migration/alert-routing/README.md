# alert-routing verifier bundle

Authored richer-asset bundle for the `alert-routing` variant in the
`normalizer-api-migration` family.

It upgrades the original easy migration task with:

- authored oracle patches for a two-step solve path
- hidden milestone slices
- follow-up coverage for lifecycle-wrapped alert titles
- red-team exploits that should still be rejected
- mutation evidence showing the hidden suite kills the known status-wrapper
  shortcuts

The latent trap is specific to alerting systems: round 1 can migrate the repo to
`RulePlan` v2 and still leave the canonical slug sensitive to transient paging
state such as severity or lifecycle wrappers. The follow-up hidden tests and
mutation checks require dispatch identity to remain stable across those wrappers.
