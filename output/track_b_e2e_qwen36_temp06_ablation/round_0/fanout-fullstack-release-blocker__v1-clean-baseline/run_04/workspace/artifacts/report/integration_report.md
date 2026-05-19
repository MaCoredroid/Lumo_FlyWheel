# Integration Report — Release-Gating Rename

## Summary

Renamed the active approval state token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation. Legacy `manual_review` is preserved as a compatibility mapping in the backend normalizer.

## Changes

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` → `{"human_review_required", "auto_approve", "blocked"}`
  - Added `LEGACY_STATE_MAP` so `"manual_review"` normalizes to `"human_review_required"`

- `services/api/src/routes/releases.py`
  - Default fallback in `build_release_gate_update` → `"human_review_required"`
  - Default fallback in `serialize_release_gate` → `"human_review_required"`
  - `operator_hint` updated to reference `"human_review_required"`

### Frontend

- `apps/admin/src/components/ReleaseGateForm.tsx`
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`

- `apps/admin/src/lib/api.ts`
  - Fallback in `buildReleaseGateRequest` → `"human_review_required"`
  - Fallback in `summarizeGateEcho` → `"human_review_required"`

### Documentation

- `docs/runbooks/release_gating.md`
  - Documents retired token (`manual_review`), new token (`human_review_required`), and verification order
  - Notes the legacy compatibility mapping

## Test Alignment

- `test_legacy_manual_review_normalizes` — passes: `"manual_review"` → `"human_review_required"`
- `test_release_gate_update_emits_new_token` — passes: update echoes `"human_review_required"`
- `test_release_gate_echo_uses_new_token` — passes: serialize echoes `"human_review_required"`
- `release-gate.spec.ts` — passes: request body sends `"human_review_required"`

## Proof Artifact

- `artifacts/dom/release_gate_capture.json` — request/echo pair confirms both sides agree on `human_review_required`
