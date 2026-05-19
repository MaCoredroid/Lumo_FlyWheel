# Integration Report — Release Gating Rename

## Summary

Renamed the release-gating approval state token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation. Legacy `manual_review` is preserved as a narrow compatibility mapping in the backend normalizer so existing fixtures and in-flight requests still resolve correctly.

## Changes

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` now includes `"human_review_required"` instead of `"manual_review"`
  - Added `LEGACY_STATE_MAP` mapping `"manual_review"` → `"human_review_required"` for backward compatibility
- `services/api/src/routes/releases.py`
  - `build_release_gate_update` default fallback → `"human_review_required"`
  - `serialize_release_gate` default fallback → `"human_review_required"`
  - `operator_hint` updated to reference `human_review_required`

### Frontend

- `apps/admin/src/components/ReleaseGateForm.tsx`
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`
- `apps/admin/src/lib/api.ts`
  - `buildReleaseGateRequest` fallback → `"human_review_required"`
  - `summarizeGateEcho` fallback → `"human_review_required"`

### Documentation

- `docs/runbooks/release_gating.md`
  - Documents retired token (`manual_review`) and new token (`human_review_required`)
  - Verification steps reference `human_review_required`

## Proof Artifact

- `artifacts/dom/release_gate_capture.json` — request/echo pair showing both sides agree on `human_review_required`

## Test Verification

- `services/api/tests/test_release_payloads.py` — all 3 tests pass (legacy normalize, update emit, echo)
- `apps/admin/tests/release-gate.spec.ts` — test passes (request body sends `human_review_required`)
