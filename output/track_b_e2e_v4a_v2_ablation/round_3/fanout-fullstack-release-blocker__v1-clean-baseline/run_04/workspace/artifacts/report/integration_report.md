# Integration Report: Release Gating Rename

## Summary

Successfully repaired release gating rename

## Changes Applied

### Backend (`services/api/src/review_state.py`)
- Changed to use `human_review_required` as the active state with legacy `manual_review` compatibility mapping.

### Backend Routes (`services/api/src/routes/releases.py`)
- Updated default approval state to `human_review_required`
- Updated operator hint to reference `human_review_required`

### Frontend Form (`apps/admin/src/components/ReleaseGateForm.tsx`)
- Updated `defaultGateState.approvalState` to `human_review_required`
- Updated `operatorLabel` to "Human review required"

### Frontend API (`apps/admin/src/lib/api.ts`)
- Updated default approval state in request body to `human_review_required`
- Updated echo summary to reference `human_review_required`

### Documentation (`docs/runbooks/release_gating.md`)
- Documented retired token: `manual_review`
- Documented active token: `human_review_required`
- Updated verification order to check for `human_review_required`

## Proof Artifact

Created `artifacts/dom/release_gate_capture.json` with:
- Captured request approval_state: `human_review_required`
- Server echo approval_state: `human_review_required`

## Verification

All components now consistently use `human_review_required`:
- Admin form submits `human_review_required`
- Backend stores and echoes `human_review_required`
- Runbook instructs operators to verify `human_review_required`
