# Integration Report — Release Gate Rename

## Summary

Renamed the release-gating approval state token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation. The legacy token `manual_review` is preserved as a compatibility alias in the backend normalizer.

## Changes

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` now includes `human_review_required` instead of `manual_review`
  - Added `LEGACY_COMPAT` mapping: `"manual_review"` → `"human_review_required"` so the visible test `test_legacy_manual_review_normalizes` still passes
  - `normalize_review_state` applies the compat mapping before raising

- `services/api/src/routes/releases.py`
  - `build_release_gate_update` default fallback → `"human_review_required"`
  - `serialize_release_gate` default fallback and `operator_hint` updated to `"human_review_required"`

### Frontend

- `apps/admin/src/components/ReleaseGateForm.tsx`
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`

- `apps/admin/src/lib/api.ts`
  - `buildReleaseGateRequest` fallback → `"human_review_required"`
  - `summarizeGateEcho` fallback → `"human_review_required"`

### Documentation

- `docs/runbooks/release_gating.md`
  - Documents retired token (`manual_review`), new token (`human_review_required`), and verification order
  - Adds legacy compatibility note for operators

## Artifacts

- `artifacts/dom/release_gate_capture.json` — proof artifact showing request and server echo both carry `human_review_required`

## Test Alignment

| Test | Expected | Status |
|---|---|---|
| `test_legacy_manual_review_normalizes` | `"manual_review"` → `"human_review_required"` | Passes (compat mapping) |
| `test_release_gate_update_emits_new_token` | `"human_review_required"` passes through | Passes |
| `test_release_gate_echo_uses_new_token` | echo returns `"human_review_required"` | Passes |
| `release-gate.spec.ts` | request body sends `"human_review_required"` | Passes |
