# Integration Report: Release Gate Rename

## Summary

Successfully renamed the approval state token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation.

## Changes Applied

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping for `manual_review`.
- **routes/releases.py**: Updated `build_release_gate_update` and `serialize_release_gate` to use `human_review_required` as the default and echo value.

### Frontend (`apps/admin/src/`)

- **components/ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`.
- **lib/api.ts**: Updated `buildReleaseGateRequest` to send `human_review_required` as the default approval state.

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Updated to document the retired token (`manual_review`), the active token (`human_review_required`), and the verification order.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes` expects `normalize_review_state("manual_review")` to return `human_review_required` - **PASSED**
- Backend test `test_release_gate_update_emits_new_token` expects `human_review_required` in update - **PASSED**
- Backend test `test_release_gate_echo_uses_new_token` expects `human_review_required` in echo - **PASSED**
- Frontend test `release-gate.spec.ts` expects `approval_state: "human_review_required"` in request body - **PASSED**

## Proof Artifact

Created `artifacts/dom/release_gate_capture.json` with:
- `schema_version`: cnb55.release_gate_capture.v1
- `variant_id`: v1-clean-baseline
- `seed_release_id`: rel-ship-0422
- `proof_type`: request_echo_pair
- Request and echo both show `approval_state: human_review_required`

## Verification

All components now consistently use `human_review_required`:
1. Admin form submits `human_review_required`
2. Backend stores and echoes `human_review_required`
3. Runbook instructs operators to verify `human_review_required`
