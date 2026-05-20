# Integration Report — Release-Gating Rename

## Summary

Renamed the release-gating approval token from `manual_review` to `human_review_required` across backend parsing, frontend submission, and operator documentation. Legacy `manual_review` survives only as a compatibility mapping in the backend normalizer.

## Changes Made

### Backend

- **`services/api/src/review_state.py`**
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` now contains `"human_review_required"` instead of `"manual_review"`
  - `normalize_review_state()` maps legacy `"manual_review"` → `"human_review_required"`

- **`services/api/src/routes/releases.py`**
  - Default fallback in `build_release_gate_update()` → `"human_review_required"`
  - Default fallback in `serialize_release_gate()` → `"human_review_required"`
  - `operator_hint` updated to reference `human_review_required`

### Frontend

- **`apps/admin/src/components/ReleaseGateForm.tsx`**
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`

- **`apps/admin/src/lib/api.ts`**
  - Fallback in `buildReleaseGateRequest()` → `"human_review_required"`
  - Fallback in `summarizeGateEcho()` → `"human_review_required"`

### Documentation

- **`docs/runbooks/release_gating.md`**
  - Documents the retired token (`manual_review`), the new token (`human_review_required`), and the verification order for operators.

## Verification

- Backend test `test_legacy_manual_review_normalizes` — `normalize_review_state("manual_review")` returns `"human_review_required"` ✓
- Backend test `test_release_gate_update_emits_new_token` — `build_release_gate_update` emits `"human_review_required"` ✓
- Backend test `test_release_gate_echo_uses_new_token` — `serialize_release_gate` echoes `"human_review_required"` ✓
- Frontend test `release-gate.spec.ts` — request body sends `approval_state: "human_review_required"` ✓

## Artifacts

- `artifacts/dom/release_gate_capture.json` — proof artifact showing seeded request payload and server echo agree on `human_review_required`
- `artifacts/report/integration_report.md` — this report
