# Workspace approvals

Operators now validate the `approval_state` column end to end across service, API, CLI, and UI.

Legacy rows without `approval_state` render as `manual_review` until the backfill completes. The UI must show `legacy_fallback` provenance for those rows.

Variant note: Baseline rollout; prove legacy fallback without extra context.

The preview artifact is `workspace-admin-approval-state.png`.
