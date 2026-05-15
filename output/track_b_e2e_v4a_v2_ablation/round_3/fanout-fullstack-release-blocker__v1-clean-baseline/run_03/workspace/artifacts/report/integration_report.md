# Integration Report: Release Gating Rename

## Summary

Successfully completed the release-gating token rename from `manual_review` to `human_review_required` across backend, frontend, and documentation.

## Changes Made

### Backend (`services/api/src/review_state.py`)
- Updated `DEFAULT_APPROVAL_STATE` to `human_review_required`
- Updated `ACTIVE_APPROVAL_STATES` to use `human_review_required`
- Added legacy compatibility mapping: `manual_review` → `human_review_required`

### Backend (`services/api/src/routes/releases.py`)
- Updated `build_release_gate_update` default to `human_review_required`
- Updated `serialize_release_gate` default to `human_review_required`
- Updated operator hint to reference `human_review_required`

### Frontend (`apps/admin/src/components/ReleaseGateForm.tsx`)
- Updated `defaultGateState.approvalState` to `human_review_required`
- Updated `operatorLabel` to `Human review required`

### Frontend (`apps/admin/src/lib/api.ts`)
- Updated `buildReleaseGateRequest` default to `human_review_required`
- Updated `summarizeGateEcho` default to `human_review_required`

### Documentation (`docs/runbooks/release_gating.md`)
- Documented token migration (retired: `manual_review`, new: `human_review_required`)
- Updated verification order to reference `human_review_required`
- Added compatibility note for legacy payloads

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json`: Captured request/echo pair showing `human_review_required` consistency

## Test Compatibility

- Backend test `test_legacy_manual_review_normalizes` expects `manual_review` → `human_review_required` mapping ✓
- Backend test `test_release_gate_update_emits_new_token` expects `human_review_required` ✓
- Backend test `test_release_gate_echo_uses_new_token` expects `human_review_required` ✓
- Frontend test `release gate request posts approval_state` expects `human_review_required` ✓

## End-to-End Verification

| Component | Token | Status |
|-----------|-------|--------|
| Admin form submission | `human_review_required` | ✓ |
| Backend normalization | `human_review_required` | ✓ |
| Server echo | `human_review_required` | ✓ |
| Runbook verification | `human_review_required` | ✓ |
