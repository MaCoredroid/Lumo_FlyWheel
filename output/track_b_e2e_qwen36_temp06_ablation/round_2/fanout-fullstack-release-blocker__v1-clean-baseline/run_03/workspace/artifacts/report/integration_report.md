# Integration Report — Release-Gating Integration Report

## Summary

Renamed the active release-gating approval token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation. Legacy `manual_review` survives as a backward-compatibility mapping in the normalizer.

## Changes Made

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` changed to `human_review_required`
  - `ACTIVE_APPROVAL_STATES` updated to include `human_review_required`
  - Added `LEGACY_STATE_MAP` to normalize `manual_review` -> `human_review_required`
  - `normalize_review_state` now applies legacy mapping before validation

- `services/api/src/routes/releases.py`
  - `build_release_gate_update` default fallback changed to `human_review_required`
  - `serialize_release_gate` default fallback and `operator_hint` updated to `human_review_required`

### Frontend

- `apps/admin/src/components/ReleaseGateForm.tsx`
  - `defaultGateState.approvalState` changed to `human_review_required`
  - `operatorLabel` updated to `Human review required`

- `apps/admin/src/lib/api.ts`
  - `buildReleaseGateRequest` fallback changed to `human_review_required`
  - `summarizeGateEcho` fallback changed to `human_review_required`

### Documentation

- `docs/runbooks/release_gating.md`
  - Documented retired token (`manual_review`) and new token (`human_review_required`)
  - Updated verification steps to reference `human_review_required`
  - Added migration notes for operators

## Proof Artifact

- `artifacts/dom/release_gate_capture.json` — request/echo pair showing `human_review_required` end-to-end

## Test Alignment

- `test_legacy_manual_review_normalizes()` — `normalize_review_state("manual_review")` returns `"human_review_required"`
- `test_release_gate_update_emits_new_token()` — `build_release_gate_update` emits `"human_review_required"`
- `test_release_gate_echo_uses_new_token()` — `serialize_release_gate` echoes `"human_review_required"`
- `release-gate.spec.ts` — `buildReleaseGateRequest` posts `"human_review_required"`
