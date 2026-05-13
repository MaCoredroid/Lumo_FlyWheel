# Release gating runbook

## Token Migration

- **Retired token:** `manual_review`
- **New token:** `human_review_required`

## Verification order:

1. Submit the release gate from the admin form with `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required` at `/api/releases/{release_id}`.
3. Confirm the operator checklist matches the same token (`human_review_required`).
