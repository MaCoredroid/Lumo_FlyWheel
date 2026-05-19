# Release gating runbook

Retired token: `manual_review`
Active token: `human_review_required`

1. Submit the release gate from the admin form — the request body must contain `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token `human_review_required`.

Note: the legacy `manual_review` token is still accepted by the backend as a compatibility mapping and will be normalized to `human_review_required`.
