# Integration Report: Release Gating Rename

## Summary

Successfully migrated the release gating token from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied Changes

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added legacy compatibility mapping for `manual_review`.
- `routes/releases.py`: Updated `build_release_gate_update()` and `serialize_release_gate()` to use `human_review_required` as default and in operator hints.

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required`.
- `lib/api.ts`: Updated `buildReleaseGateRequest()` and `summarizeGateEcho()` to use `human_review_required`.

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Documented the token rename with retired and active tokens, and updated verification order.

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Contains request_echo_pair proof showing both captured request and server echo using `human_review_required`.

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes()` passes: `manual_review` maps to `human_review_required`.
- Backend test `test_release_gate_update_emits_new_token()` passes: update emits `human_review_required`.
- Backend test `test_release_gate_echo_uses_new_token()` passes: echo uses `human_review_required`.
- Frontend test `release gate request posts approval_state` passes: request body contains `human_review_required`.

## End-to-End Verification

| Component | Token | Status |
|-----------|-------|--------|
| Admin form submission | `human_review_required` | ✓ |
| Backend storage | `human_review_required` | ✓ |
| Server echo | `human_review_required` | ✓ |
| Runbook documentation | `human_review_required` | ✓ |
