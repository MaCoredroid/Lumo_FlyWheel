# Integration Report: Release Gating Token Migration

## Summary

Migrated the release-gating approval state token from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied Changes

### Backend (`services/api/src/`)

- `review_state.py`:
  - Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`
  - Updated `ACTIVE_APPROVAL_STATES` to use `human_review_required`
  - Added compatibility mapping: `manual_review` normalizes to `human_review_required`

- `routes/releases.py`:
  - Updated default approval state in `build_release_gate_update` to `human_review_required`
  - Updated default in `serialize_release_gate` to `human_review_required`
  - Updated operator hint to reference `human_review_required`

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`:
  - Updated `defaultGateState.approvalState` to `human_review_required`
  - Updated `operatorLabel` to `Human review required`

- `lib/api.ts`:
  - Updated default fallback in `buildReleaseGateRequest` to `human_review_required`
  - Updated fallback in `summarizeGateEcho` to `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`:
  - Documented retired token (`manual_review`) and active token (`human_review_required`)
  - Updated verification order to reference `human_review_required`

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes`: Passes (maps `manual_review` to `human_review_required`)
- Backend test `test_release_gate_update_emits_new_token`: Passes (uses `human_review_required`)
- Backend test `test_release_gate_echo_uses_new_token`: Passes (uses `human_review_required`)
- Frontend test `release-gate.spec.ts`: Passes (uses `human_review_required`)

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Contains request/echo pair proof with `human_review_required`

## Verification

End-to-end flow is coherent:
1. Admin form submits `human_review_required`
2. Backend stores and echoes `human_review_required`
3. Runbook instructs operators to verify `human_review_required`
