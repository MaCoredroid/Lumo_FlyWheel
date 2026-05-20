# Release gating runbook

**Token rename (retired → active):** `manual_review` → `human_review_required`

1. Submit the release gate from the admin form — the request body must contain `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token (`human_review_required`).

> **Note:** Legacy fixtures and historical data may still reference `manual_review`. The backend normalizes `manual_review` to `human_review_required` for compatibility. Operators should verify the active token `human_review_required` in all new releases.
