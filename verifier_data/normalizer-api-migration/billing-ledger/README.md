# billing-ledger verifier bundle

Authored richer-asset bundle for the `billing-ledger` variant in the
`normalizer-api-migration` family.

It upgrades the original easy migration task with:

- authored oracle patches for a two-step solve path
- hidden milestone slices
- follow-up coverage for separator-heavy ledger titles
- red-team exploits that should still be rejected
- mutation evidence showing the hidden suite kills the known normalization
  shortcuts

The latent trap is specific to billing exports: round 1 can migrate the repo to
`RulePlan` v2 and still leave title normalization too naive for strings such as
`Refund / Retry _ Queue` or `Chargeback---Retry!!!`. The follow-up hidden tests
and mutation checks require canonical slugs with single hyphen separators.
