# Release gating runbook

> **Token rename (2026-05-20):** `manual_review` was retired and replaced with
> `human_review_required`. The backend still accepts `manual_review` as a
> legacy alias and maps it to `human_review_required`, but all new submissions
> and operator checks must use the new token.

## Verification order

1. Submit the release gate from the admin form — confirm the request body
   contains `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token
   (`human_review_required`), not the retired `manual_review`.
