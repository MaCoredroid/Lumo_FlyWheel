# Integration Report: Release Gating Rename

## Summary

Successfully migrated release gating from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added legacy mapping for `manual_review`
- `routes/releases.py`: Updated default state and operator hint to use `human_review_required`

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required`
- `lib/api.ts`: Updated default approval_state in request body to `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Documented retired token (`manual_review`), active token (`human_review_required`), and verification order

## Compatibility

Legacy `manual_review` values are normalized to `human_review_required` via `normalize_review_state()` to ensure backward compatibility with existing fixtures and tests.

## Test Alignment

All visible tests pass with the new token:
- `test_legacy_manual_review_normalizes`: Confirms legacy mapping
- `test_release_gate_update_emits_new_token`: Confirms request body
- `test_release_gate_echo_uses_new_token`: Confirms server echo
- `release-gate.spec.ts`: Confirms frontend request
