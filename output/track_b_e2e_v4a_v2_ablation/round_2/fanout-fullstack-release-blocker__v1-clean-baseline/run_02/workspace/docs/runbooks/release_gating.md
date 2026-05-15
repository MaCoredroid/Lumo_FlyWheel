# Release gating runbook

**Token rename**: `manual_review` → `human_review_required`

## Verification order

1. Submit the release gate from the admin form with `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token.

## Legacy compatibility

The backend accepts `manual_review` as a legacy token and normalizes it to `human_review_required`.
