# Integration Report: Release Gating Token Rename

## Summary

Successfully migrated release gating from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`. Added legacy compatibility mapping for `manual_review` -> `human_review_required`.
- **routes/releases.py**: Updated `build_release_gate_update` and `serialize_release_gate` now default to and echo `human_review_required`.

### Frontend (`apps/admin/src/`)

- **components/`)

- **ReleaseGateForm.tsx**: Changed `defaultGateState.approvalState` to `human_review_required`.
- **lib/api.tsbuildReleaseGateRequest` now defaults to `human_review_required`.

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Documented the token migration with retired token (`manual_review`), new token (`human_review_required`), and verification order.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` verifies `manual_review` normalizes to `human_review_required`.
- Frontend test `release-gate.spec.ts` verifies request body sends `human_review_required`.

## Proof Artifacts

- **artifacts/dom/release_gate_capture.json**: DOM/request proof showing request and server echo with `human_review_required`.
