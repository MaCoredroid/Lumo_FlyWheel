# Release gating runbook

**Token rename:** `manual_review` (retired) -> `human_review_required` (active).

1. Submit the release gate from the admin form.
2. Watch the server echo for `human_review_required`.
3. Confirm the operator checklist matches `human_review_required`.

## Migration notes

- Legacy payloads containing `manual_review` are normalized to `human_review_required` by the backend.
- Operators should verify the new token `human_review_required` in all new deployments.
- The retired token `manual_review` survives only as a backward-compatibility mapping in the normalizer.
