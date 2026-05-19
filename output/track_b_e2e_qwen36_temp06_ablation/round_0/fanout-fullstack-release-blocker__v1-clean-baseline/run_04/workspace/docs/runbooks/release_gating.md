# Release gating runbook

Retired token: `manual_review` (legacy). New active token: `human_review_required`.

1. Submit the release gate from the admin form — the request body must contain `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required` in the response.
3. Confirm the operator checklist matches the same token (`human_review_required`).

Note: the backend still accepts `manual_review` as a legacy compatibility mapping, normalizing it to `human_review_required`. Operators should only use the new token going forward.
