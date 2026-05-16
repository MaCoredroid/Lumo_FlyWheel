# Integration Report: Release Gate Token Migration

## Summary

Successfully migrated the release gating system from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (services/api)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping for `manual_review`.
- **routes/releases.py**: Updated `build_release_gate_update` and `serialize_release_gate` to use `human_review_required` as default and in operator hints.

### Frontend (apps/admin)

- **ReleaseGateForm.tsx**: Updated `defaultGateState` to use `human_review_required` as the approval state.
- **api.ts**: Updated `buildReleaseGateRequest` to send `human_review_required` as the default approval state.

### Documentation

- **release_gating.md**: Updated runbook to document the token migration (retired: `manual_review`, active: `human_review_required`) and verification order.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes`: Passes - `manual_review` normalizes to `human_review_required`
- Backend test `test_release_gate_update_emits_new_token`: Passes - emits `human_review_required`
- Backend test `test_release_gate_echo_uses_new_token`: Passes - echoes `human_review_required`
- Frontend test `release gate request posts approval_state`: Passes - request body contains `human_review_required`

## Proof Artifact

- **artifacts/dom/release_gate_capture.json**: Contains request/echo pair proof with `human_review_required` token in both captured request and server echo.

## Verification

All components now consistently use `human_review_required` as the active approval state token. The legacy `manual_review` token survives only as a narrow compatibility mapping in the backend normalizer.
