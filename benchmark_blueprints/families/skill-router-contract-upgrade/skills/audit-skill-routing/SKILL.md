# Audit Skill Routing

Use this skill when a repo-local skill router is mid-migration from a single-trigger schema to a richer routing contract.

## Do

- Identify every field that now influences eligibility: `triggers`, `negative_triggers`, and `required_inputs`
- Treat suppressors and missing inputs as first-class routing gates
- Verify fallback only runs when no eligible skill remains
- Keep docs and config examples on the live schema

## Do Not

- Do not stop after adding list support for `triggers`
- Do not encode a blanket default fallback
- Do not delete ambiguous fixtures to simplify the problem

## Working Pattern

1. Determine how eligibility is computed.
2. Audit how ranking behaves after suppression and input gating.
3. Align docs and config examples with the live contract.
4. Re-check ambiguous prompts rather than only obvious happy paths.

## Success Signal

The router chooses the correct skill under ambiguous prompts, only falls back when appropriate, and the docs/config use the live schema keys.
