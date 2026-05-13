# Integration Report: Release Gating Rename

## Summary

Successfully migrated release gating from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added legacy compatibility mapping for `manual_review`
- `routes/releases.py`: Updated default approval state and operator hint to use `human_review_required`

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required`
- `lib/api.ts`: Updated request builder default and echo summary to use `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Documented retired token (`manual_review`), active token (`human_review_required`), and verification order

## Test Alignment

- `services/api/tests/test_release_payloads.py`: Tests expect `human_review_required` with legacy `manual_review` mapping
- `apps/admin/tests/release-gate.spec.ts`: Tests expect request body to contain `human_review_required`

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Captured request and server echo proof with matching `human_review_required` tokens

## Verification

All components now coherently use `human_review_required` as the active approval state token.
