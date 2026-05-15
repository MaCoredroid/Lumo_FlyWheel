# Integration Report: Release Gate Token Rename

## Summary

Successfully migrated release gating from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/`)

- `review_state.py`:
  - Changed `DEFAULT_APPROVAL_STATE` from `manual_review` to `human_review_required`
  - Updated `ACTIVE_APPROVAL_STATES` to use `human_review_required`
  - Added legacy compatibility mapping: `manual_review` → `human_review_required`

- `routes/releases.py`:
  - Updated default approval state to `human_review_required`
  - Updated `operator_hint` to reference `human_review_required`

### Frontend (`apps/admin/src/`)

- `components/`)

- `ReleaseGateForm.tsx`:
  - Changed `defaultGateState.approvalState` to `human_review_required`
  - Updated `operatorLabel` to "Human review required"

- `lib/api.ts`:
  - Updated default fallback to `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`:
  - Documented retired token (`manual_review`)
  - Documented active token (`human_review_required`)
  - Updated verification order to reference new token

## Test Alignment

All changes align with existing tests:
- `services/api/tests/test_release_payloads.py` expects `human_review_required`
- `apps/admin/tests/release-gate.spec.ts` expects `human_review_required`

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: DOM/request proof showing request and echo agreement
