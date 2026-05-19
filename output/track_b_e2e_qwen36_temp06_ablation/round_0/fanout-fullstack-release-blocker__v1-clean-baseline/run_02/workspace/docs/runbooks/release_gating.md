# Release gating runbook

**Token rename**: retired `manual_review` → active `human_review_required`.

1. Submit the release gate from the admin form — the request body must contain `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches `human_review_required`.

**Legacy note**: The backend still accepts `manual_review` as a compatibility alias that normalizes to `human_review_required`. Operators should only use `human_review_required` going forward.
