# Release gating runbook

## Token Migration

- Retired token: `manual_review`
- New token: `human_review_required`

## Verification Order

1. Submit the release gate from the admin form.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token.
