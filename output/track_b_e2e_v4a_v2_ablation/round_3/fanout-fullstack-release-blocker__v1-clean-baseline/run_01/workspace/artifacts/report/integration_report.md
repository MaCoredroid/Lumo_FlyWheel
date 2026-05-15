# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release gating approval state from `manual_review` to `human_review_required` across all layers.

## Changes Made

### Backend (`services/api/src/`)

- **review_state.py**: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added legacy compatibility mapping for `manual_review` -> `human_review_required`
- **releases.py**: Updated default values and operator hint to reference `human_review_required`

### Frontend (`apps/admin/src/`)

- **ReleaseGateForm.tsx**: Updated `defaultGateState.approvalState` to `human_review_required`
- **api.ts**: Updated default approval state in `buildReleaseGateRequest` and `summarizeGateEcho`

### Documentation (`docs/runbooks/`)

- **release_gating.md**: Documented token migration (retired: `manual_review`, new: `human_review_required`) and updated verification order

## Test Alignment

- `test_legacy_manual_review_normalizes`: Verifies `manual_review` maps to `human_review_required`
- `test_release_gate_update_emits_new_token`: Verifies update emits `human_review_required`
- `test_release_gate_echo_uses_new_token`: Verifies echo uses `human_review_required`
- `release-gate.spec.ts`: Verifies frontend request posts `human_review_required`

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Request/echo pair proof with `human_review_required`

## Verification

All code paths now use `human_review_required` as the active approval state token. Legacy `manual_review` is preserved only as a compatibility mapping in `normalize_review_state()`.
