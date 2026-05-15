# Release gating runbook

## Migration Summary

- **Retired token**: `manual_review`
- **New token**: `human_review_required`

The legacy `manual_review` token is now mapped to `human_review_required` in the backend for compatibility.

## Verification Order

1. Submit the release gate from the admin form.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches the same token (`human_review_required`).
4. Verify the database stores `human_review_required` as the approval state.

## Compatibility Note

Legacy payloads with `approval_state: "manual_review"` are automatically normalized to `human_review_required` by the backend.
