# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release-gating token from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping so `manual_review` normalizes to `human_review_required`.
- **routes/releases.py**: Updated `build_release_gate_update()` and `serialize_release_gate()` to use `human_review_required` as the default and in operator hints.

### Frontend (`apps/admin/src/`)

- **components/ReleaseGateForm.tsx**: Updated `defaultGateState` to use `approvalState: "human_review_required"` and updated `operatorLabel`.
- **lib/api.ts**: Updated `buildReleaseGateRequest()` to default to `human_review_required` and updated `summarizeGateEcho()` message.

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Updated runbook to document the token migration (retired: `manual_review`, active: `human_review_required`) and updated verification order to reference the new token.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` passes: `manual_review` normalizes to `human_review_required`.
- Backend test `test_release_gate_update_emits_new_token()` passes: update emits `human_review_required`.
- Backend test `test_release_gate_echo_uses_new_token()` passes: echo uses `human_review_required`.
- Frontend test `release-gate.spec.ts` passes: request body contains `human_review_required`.

## Proof Artifact

- **artifacts/dom/release_gate_capture.json**: Captured request and server echo both show `human_review_required` for release `rel-ship-0422`.

## Verification Checklist

- [x] Backend active state is `human_review_required`
- [x] Legacy `manual_review` survives as compatibility mapping
- [x] Frontend request body sends `approval_state: "human_review_required"`
- [x] Runbook names retired token, new token, and verification order
- [x] Proof artifact shows seeded request and server echo agreeing
