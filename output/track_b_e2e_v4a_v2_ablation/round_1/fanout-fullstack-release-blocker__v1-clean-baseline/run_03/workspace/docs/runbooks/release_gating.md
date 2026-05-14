# Release gating runbook

## Token migration

- **Retired token**: `manual_review`
- **Active token**: `human_review_required`

## Verification order

1. Submit the release gate from the admin form with `human_review_required`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches `human_review_required`.
