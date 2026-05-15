# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release-gating token from `manual_review` to `human_review_required` across backend parsing, frontend submission, and operator documentation.

## Changes Made

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added legacy compatibility mapping for `manual_review`
- `releases.py`: Updated default approval state and operator hint to reference `human_review_required`

### Frontend (`apps/admin/src/`)

- `ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required`
- `api.ts`: Updated default `approval_state` in request builder to `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Documented retired token, new token, and verification order

## Test Compatibility

- `test_legacy_manual_review_normalizes()`: `manual_review` normalizes to `human_review_required` ✓
- `test_release_gate_update_emits_new_token()`: Backend emits `human_review_required` ✓
- `test_release_gate_echo_uses_new_token()`: Serialization uses `human_review_required` ✓
- Frontend test expects `human_review_required` in request body ✓

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Captured request and server echo both show `human_review_required` ✓

## End-to-End Verification

The admin form submits `human_review_required`, the backend stores and echoes `human_review_required`, and the runbook tells operators to verify `human_review_required`.
