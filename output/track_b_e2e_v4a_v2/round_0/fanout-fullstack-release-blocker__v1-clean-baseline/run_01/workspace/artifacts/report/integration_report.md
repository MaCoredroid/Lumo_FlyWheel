# Integration Report: Release Gating Rename

## Summary

Successfully migrated release gating from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping for `manual_review` -> `human_review_required`.
- **routes/releases.py**: Updated default approval state and operator hint to reference `human_review_required`.

### Frontend (`apps/admin/src/`)

- **components/`)

- **ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`.

### Frontend (`apps/admin/src/lib/`)

- **api.ts**: Updated default approval state in `buildReleaseGateRequest` and `summarizeGateEcho` to `human_review_required`.

### Documentation updated.

## Verification

- Backend test `test_legacy_manual_review_normalizes` confirms `manual_review` maps to `human_review_required`.
- Frontend test `release gate request posts approval_state` confirms request body sends `human_review_required`.
- Proof artifact at `artifacts/dom/release_gate_capture.json` documents request/echo pair with matching token.

## Conclusion

End-to-end coherence achieved: admin form, backend storage, and operator documentation all use `human_review_required`.
