# Integration Report — Release-Gating Rename

**Date:** 2026-05-20
**Variant:** v1-clean-baseline

## Summary

Renamed the release-gating approval token from `manual_review` (retired) to `human_review_required` (current) across backend, frontend, and operator documentation.

## Changes

### Backend

- **`services/api/src/review_state.py`**
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` → `{"human_review_required", "auto_approve", "blocked"}`
  - `normalize_review_state` now maps legacy `"manual_review"` → `"human_review_required"` via `_COMPAT_MAP`

- **`services/api/src/routes/releases.py`**
  - `build_release_gate_update` default fallback → `"human_review_required"`
  - `serialize_release_gate` default fallback → `"human_review_required"`
  - `operator_hint` updated to reference `human_review_required`

### Frontend

- **`apps/admin/src/components/ReleaseGateForm.tsx`**
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`

- **`apps/admin/src/lib/api.ts`**
  - `buildReleaseGateRequest` fallback → `"human_review_required"`
  - `summarizeGateEcho` fallback → `"human_review_required"`

### Documentation

- **`docs/runbooks/release_gating.md`**
  - Added token-rename note and updated verification steps to reference `human_review_required`.

## Test Compatibility

- `services/api/tests/test_release_payloads.py` (read-only): expects `normalize_review_state("manual_review") == "human_review_required"` — satisfied by `_COMPAT_MAP`.
- `apps/admin/tests/release-gate.spec.ts` (read-only): expects request body `approval_state` to be `"human_review_required"` — satisfied by updated default.

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json` — request/echo pair showing `human_review_required` end-to-end.
