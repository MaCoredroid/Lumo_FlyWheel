# Integration Report: Release Gating Token Migration

## Summary

Successfully migrated the release gating system from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added `LEGACY_STATE_MAPPINGS` for backward compatibility with `manual_review`
- **routes/releases.py**: Updated default approval state and operator hint to reference `human_review_required`

### Frontend (`apps/admin/src/`)

- **components/ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`
- **lib/api.ts**: Updated default fallback and echo summary to use `human_review_required`

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Added token migration section documenting retired/new tokens and updated verification steps

## Proof Artifacts

- **artifacts/dom/release_gate_capture.json**: Contains request/echo pair proof showing `human_review_required` token end-to-end

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` verifies `manual_review` maps to `human_review_required`
- Frontend test `release-gate.spec.ts` verifies request body sends `human_review_required`

## Verification Status

- [x] Backend active state is `human_review_required`
- [x] Legacy `manual_review` survives as compatibility mapping
- [x] Frontend request body sends `approval_state: "human_review_required"`
- [x] Runbook names retired token, new token, and verification order
- [x] Proof artifact shows seeded request payload and server echo agreeing
