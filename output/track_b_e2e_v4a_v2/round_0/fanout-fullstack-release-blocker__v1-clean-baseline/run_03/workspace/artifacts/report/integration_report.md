# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release gating system from manual_review to human_review_required across backend, frontend, and documentation.

## Changes Applied

### Backend (services/api/src/)

- review_state.py: Updated DEFAULT_APPROVAL_STATE and ACTIVE_APPROVAL_STATES to use human_review_required. Added legacy compatibility mapping for manual_review to human_review_required.
- routes/releases.py: Updated default values and operator hints to reference human_review_required.

### Frontend (apps/admin/src/)

- ReleaseGateForm.tsx: Updated defaultGateState.approvalState to human_review_required.
- lib/api.ts: Updated default fallback and echo summary to use human_review_required.

### Documentation (docs/runbooks/)

- release_gating.md: Added token rename note, updated verification steps to reference human_review_required.

## Proof Artifacts

- artifacts/dom/release_gate_capture.json: Contains the request/echo pair proof with schema cnb55.release_gate_capture.v1.

## Test Compatibility

- Backend test test_legacy_manual_review_normalizes() expects manual_review to normalize to human_review_required - PASSED
- Backend test test_release_gate_update_emits_new_token() expects human_review_required in update - PASSED
- Backend test test_release_gate_echo_uses_new_token() expects human_review_required in echo - PASSED
- Frontend test release gate request posts approval_state expects human_review_required in request body - PASSED

## Verification Order

1. Admin form submits approval_state: human_review_required
2. Backend normalizes and stores human_review_required
3. Server echoes human_review_required
4. Operator runbook verifies human_review_required
