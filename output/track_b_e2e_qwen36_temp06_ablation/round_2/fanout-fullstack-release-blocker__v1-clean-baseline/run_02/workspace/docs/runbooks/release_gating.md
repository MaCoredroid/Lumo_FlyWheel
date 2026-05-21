# Release gating runbook

Token Renamed

**Retired token:** `manual_review`
**New token:** `human_review_required`

The `manual_review` token was retired and replaced with `human_review_required` across the admin form, backend API, and operator tooling. The backend still accepts `manual_review` as a legacy compatibility mapping, normalizing it to `human_review_required`.

## Verification order

1. Submit the release gate from the admin form — the form now sends `approval_state: "human_review_required"`.
2. Watch the server echo for `human_review_required` (not the old `manual_review`).
3. Confirm the operator checklist matches the new token `human_review_required`.
