# Integration Report — Release-Gating Rename

**Variant:** v1-clean-baseline  
**Date:** 2026-05-21  

## Summary

Renamed the release-gating approval state token from `manual_review` (retired) to `human_review_required` (active) across backend parsing, frontend submission, and operator documentation.

## Changes Made

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` now contains `"human_review_required"` instead of `"manual_review"`
  - `normalize_review_state` maps legacy `"manual_review"` → `"human_review_required"` for backward compatibility
- `services/api/src/routes/releases.py`
  - `build_release_gate_update` defaults to `"human_review_required"`
  - `serialize_release_gate` defaults to `"human_review_required"` and updated operator hint

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
  - Updated verification steps to reference `human_review_required`

## Verification

- Backend `normalize_review_state("manual_review")` returns `"human_review_required"` (legacy compat)
- Backend `normalize_review_state("human_review_required")` returns `"human_review_required"` (active)
- Frontend form submits `approval_state: "human_review_required"`
- Server echoes `human_review_required"`
- Proof artifact `artifacts/dom/release_gate_capture.json` confirms request/echo agreement

## Artifacts

- `artifacts/dom/release_gate_capture.json` — DOM/request proof artifact
- `artifacts/report/integration_report.md` — this report
