# Release Gating Integration Report

## Summary

Successfully repaired the release-gating rename across backend, frontend, and documentation.

## Changes Applied changes

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`, added `human_review_required` to `ACTIVE_APPROVAL_STATES`, and implemented legacy compatibility mapping for `manual_review` -> `human_review_required`.

- `routes/releases.py`: Updated `build_release_gate_update()` and `serialize_release_gate()` to use `human_review_required` as the default and echo token.

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required`.

- `lib/api.ts`: Updated `buildReleaseGateRequest()` and `summarizeGateEcho()` to use `human_review_required`.

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Updated runbook to document the retired token (`manual_review`), the new token (`human_review_required`), and the verification order.

### Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Created proof artifact showing request and server echo both using `human_review_required`.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` expects `manual_review` to normalize to `human_review_required` - compatibility mapping implemented.

- Backend tests `test_release_gate_update_emits_new_token()` and `test_release_gate_echo_uses_new_token()` expect `human_review_required` - changes aligned.

- Frontend test `release-gate.spec.ts` expects `approval_state: "human_review_required"` in request body - changes aligned.

## Verification

All components now consistently use `human_review_required` as the active token. Legacy `manual_review` is preserved only as a compatibility mapping in the backend normalizer.
