# Release Gating Integration Report

**Date**: 2026-05-15
**Variant**: v1-clean-baseline

## Summary

Successfully completed the release-gating rename from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Applied

### Backend (`services/api/src/`)

- `review_state.py`:
  - Changed `DEFAULT_APPROVAL_STATE` from `manual_review` to `human_review_required`
  - Updated `ACTIVE_APPROVAL_STATES` to include `human_review_required`
  - Added legacy compatibility mapping: `manual_review` → `human_review_required`

- `routes/releases.py`:
  - Updated `build_release_gate_update` default to `human_review_required`
  - Updated `serialize_release_gate` operator_hint now references `human_review_required`

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`:
  - Changed `defaultGateState.approvalState` to `human_review_required`
  - Updated `operatorLabel` to `Human review required`

- `lib/api.ts`:
  - Updated `buildReleaseGateRequest` default to `human_review_required`
  - Updated `summarizeGateEcho` default to `human_review_required`

### Documentation (`docs/runbooks/`)

- `release_gating.md`:
  - Documented token rename: `manual_review` → `human_review_required`
  - Updated verification steps to reference new token
  - Added legacy compatibility note

## Test Alignment

- Backend test `test_legacy_manual_review_normalizes`: Expects `manual_review` to normalize to `human_review_required` ✓
- Backend test `test_release_gate_update_emits_new_token`: Expects `human_review_required` ✓
- Backend test `test_release_gate_echo_uses_new_token`: Expects `human_review_required` ✓
- Frontend test `release-gate.spec.ts`: Expects request body to contain `human_review_required` ✓

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Contains request/echo pair showing `human_review_required` end-to-end

## Verification

The integration is complete and coherent:
1. Admin form submits `human_review_required`
2. Backend stores and echoes `human_review_required`
3. Runbook instructs operators to verify `human_review_required`
4. Legacy `manual_review` survives as compatibility mapping
