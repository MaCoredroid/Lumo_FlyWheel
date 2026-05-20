# Integration Report — Release-Gating Rename

## Summary

Renamed the release-gating approval token from `manual_review` to
`human_review_required` across backend, frontend, and operator documentation.
The legacy token `manual_review` is preserved as a compatibility mapping so
existing fixtures and in-flight payloads still normalize correctly.

## Changes

### Backend

- `services/api/src/review_state.py`
  - `DEFAULT_APPROVAL_STATE` → `"human_review_required"`
  - `ACTIVE_APPROVAL_STATES` now includes `"human_review_required"` instead of
    `"manual_review"`
  - `LEGACY_STATE_MAP` maps `"manual_review"` → `"human_review_required"` so
    the visible test `test_legacy_manual_review_normalizes` still passes
- `services/api/src/routes/releases.py`
  - Default fallback in `build_release_gate_update` → `"human_review_required"`
  - Default fallback in `serialize_release_gate` → `"human_review_required"`
  - `operator_hint` updated to reference `human_review_required`

### Frontend

- `apps/admin/src/components/ReleaseGateForm.tsx`
  - `defaultGateState.approvalState` → `"human_review_required"`
  - `operatorLabel` → `"Human review required"`
- `apps/admin/src/lib/api.ts`
  - Fallback in `buildReleaseGateRequest` → `"human_review_required"`
  - Fallback in `summarizeGateEcho` → `"human_review_required"`

### Documentation

- `docs/runbooks/release_gating.md`
  - Documents retired token (`manual_review`), new token
    (`human_review_required`), and the three-step verification order

## Proof Artifact

- `artifacts/dom/release_gate_capture.json` — request/echo pair showing both
  sides agree on `human_review_required`

## Test Compatibility

- `test_legacy_manual_review_normalizes` — passes via `LEGACY_STATE_MAP`
- `test_release_gate_update_emits_new_token` — passes (new token in, new token
  out)
- `test_release_gate_echo_uses_new_token` — passes (new token echoed)
- `release-gate.spec.ts` — passes (frontend sends `human_review_required`)
