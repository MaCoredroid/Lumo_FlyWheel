# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release gating approval state from `manual_review` to `human_review_required` across backend, frontend, and operator documentation.

## Changes Made

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added `human_review_required` to `ACTIVE_APPROVAL_STATES`, and implemented legacy compatibility mapping for `manual_review`.
- **routes/releases.py**: Updated `build_release_gate_update` and `serialize_release_gate` to use `human_review_required` as default; updated operator hint.

### Frontend (`apps/admin/src/`)

- **components/ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required` and `operatorLabel` to "Human review required".
- **lib/api.ts**: Updated `buildReleaseGateRequest` default to `human_review_required` and `summarizeGateEcho` fallback.

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Document now references the retired token (`manual_review`) and active token (`human_review_required`), with verification steps for operators.

## Verification

- Backend test `test_legacy_manual_review_normalizes` expects `manual_review` to normalize to `human_review_required`.
- Frontend test `release gate request posts approval_state` expects `human_review_required` in request body.
- Proof artifact `artifacts/dom/release_gate_capture.json` captures request and server echo agreeing on `human_review_required`.

## Conclusion

The release gating rename is complete and coherent end-to-end.
