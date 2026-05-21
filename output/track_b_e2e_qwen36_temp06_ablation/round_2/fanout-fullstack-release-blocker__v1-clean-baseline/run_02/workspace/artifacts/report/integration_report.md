# Integration Report — Release Gate Rename

## Summary

Renamed the release-gating approval state token from `manual_review` to `human_review_required` across the backend, frontend, and operator documentation.

## Changes Made

### Backend

- **`services/api/src/review_state.py`**
  - `DEFAULT_APPROVAL_STATE` changed to `"human_review_required"`.
  - `ACTIVE_APPROVAL_STATES` now contains `"human_review_required"` instead of `"manual_review"`.
  - Added `LEGACY_STATE_MAP` to normalize `"manual_review"` → `"human_review_required"` for backward compatibility.

- **`services/api/src/routes/releases.py`**
  - Default `approval_state` fallback updated to `"human_review_required"`.
  - `operator_hint` updated to reference `human_review_required`.

### Frontend

- **`apps/admin/src/components/ReleaseGateForm.tsx`**
  - `defaultGateState.approvalState` set to `"human_review_required"`.
  - `operatorLabel` updated to `"Human review required"`.

- **`apps/admin/src/lib/api.ts`**
  - Fallback `approval_state` in request body and echo summary updated to `"human_review_required"`.

### Documentation

- **`docs/runbooks/release_gating.md`**
  - Documents retired token (`manual_review`) and new token (`human_review_required`).
  - Updated verification order to check for `human_review_required`.

## Test Compatibility

- **Backend tests** (`services/api/tests/test_release_payloads.py`): Expect `normalize_review_state("manual_review")` to return `"human_review_required"` — satisfied by `LEGACY_STATE_MAP`.
- **Frontend tests** (`apps/admin/tests/release-gate.spec.ts`): Expect request body to contain `approval_state: "human_review_required"` — satisfied by updated form and API defaults.

## Proof Artifact

`artifacts/dom/release_gate_capture.json` confirms the seeded request payload and server echo both carry `approval_state: "human_review_required"`.
