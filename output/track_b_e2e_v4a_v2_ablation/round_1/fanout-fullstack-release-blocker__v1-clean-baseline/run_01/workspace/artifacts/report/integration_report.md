# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release gating token from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping for `manual_review` -> `human_review_required`.
- **routes/releases.py**: Updated default values and operator hint to reference `human_review_required`.

### Frontend (`apps/admin/src/`)

- **components/`)

- **ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`.
- **lib/api.ts**: Updated default fallback value in `buildReleaseGateRequest` and `summarizeGateEcho`.

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Added token migration section documenting retired and new tokens. Updated verification order to reference `human_review_required`.

## Proof Artifacts

- **artifacts/dom/release_gate_capture.json**: Contains request/echo pair proof showing `human_review_required` in both captured request and server echo.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` verifies legacy `manual_review` maps to new token.
- Backend test `test_release_gate_update_emits_new_token()` verifies new token in update payload.
- Backend test `test_release_gate_echo_uses_new_token()` verifies new token in serialized output.
- Frontend test `release-gate.spec.ts` verifies request body contains `human_review_required`.

## Verification

All code changes align with visible test expectations. The release gating system now uses `human_review_required` as the active token while maintaining backward compatibility for legacy `manual_review` inputs.
