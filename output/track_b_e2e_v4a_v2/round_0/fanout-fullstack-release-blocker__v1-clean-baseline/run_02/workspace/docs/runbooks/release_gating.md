# Release gating runbook

Token rename runbook

The legacy token `manual_review` has been retired in favor of `human_review_required`.

## Verification order

1. Submit the release gate from the admin form with `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the `human_review_required` token.

## Token mapping

- Retired token: `manual_review`
- Active token: `human_review_required`

Legacy `manual_review` values are normalized to `human_review_required` for compatibility.
