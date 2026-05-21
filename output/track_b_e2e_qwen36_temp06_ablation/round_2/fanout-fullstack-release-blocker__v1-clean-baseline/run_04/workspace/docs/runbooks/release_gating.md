# Release gating runbook

Token rename: `manual_review` (retired) → `human_review_required` (active).

1. Submit the release gate from the admin form with `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token `human_review_required`.

Note: the backend still accepts the legacy `manual_review` token and maps it to
`human_review_required` for backward compatibility. Operators should always
verify the new token in the echo response.
