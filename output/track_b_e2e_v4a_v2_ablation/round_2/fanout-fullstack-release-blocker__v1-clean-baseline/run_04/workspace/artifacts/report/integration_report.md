# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release-gating token from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added legacy compatibility mapping for `manual_review` -> `human_review_required`
- **routes/releases.py**: Updated default approval state to `human_review_required`, updated operator hint

### Frontend (`apps/admin/src/`)

- **components/ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`
- **lib/api.ts**: Updated default fallback to `human_review_required`

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Documented retired token, new token, and verification order

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` verifies `manual_review` maps to `human_review_required`
- Backend tests verify new token is emitted and echoed correctly
- Frontend test verifies request body sends `human_review_required`

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Captured request/echo pair showing `human_review_required` token

## Verification

All components now coherently use `human_review_required` as the active approval state token.
