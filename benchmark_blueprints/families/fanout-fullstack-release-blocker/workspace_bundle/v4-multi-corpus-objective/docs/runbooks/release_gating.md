# Release gating runbook

Current rollout note still refers to `manual_review`.

1. Submit the release gate from the admin form.
2. Watch the server echo for `manual_review`.
3. Confirm the operator checklist matches the same token.

Keep the old operator order: read the server echo first and only then inspect the UI payload.
