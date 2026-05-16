# Integration Report: Release Gating Rename

## Summary

The release-gating token has been renamed from `manual_review` to `human_review_required` across backend parsing, frontend submission, and operator documentation.

## Changes Made

### Backend (`services/api/src/`)

- `review_state.py`: Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`. Added legacy compatibility mapping for `manual_review` to normalize to `human_review_required`.
- `routes/releases.py`: Updated `build_release_gate_update` and `serialize_release_gate` to use `human_review_required` as the default and echo value. Updated `operator_hint` to reference the new token.

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: Updated `defaultGateState.approvalState` to `human_review_required` and `operatorLabel` to `Human review required`.
- `lib/api.ts`: Updated `buildReleaseGateRequest` to send `human_review_required` as default. Updated `summarizeGateEcho` to reference the new token.

### Documentation (`docs/runbooks/`)

- `release_gating.md`: Updated to document the retired token (`manual_review`), new token (`human_review_required`), and verification order.

## Verification

- Backend test `test_legacy_manual_review_normalizes` verifies `manual_review` maps to `human_review_required`
- Backend test `test_release_gate_update_emits_new_token` verifies the update uses `human_review_required`
- Backend test `test_release_gate_echo_uses_new_token` verifies the echo uses `human_review_required`
- Frontend test `release-gate.spec.ts` verifies the request body sends `human_review_required`

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Contains the request/echo pair proof showing `human_review_required` in both captured request and server echo.
