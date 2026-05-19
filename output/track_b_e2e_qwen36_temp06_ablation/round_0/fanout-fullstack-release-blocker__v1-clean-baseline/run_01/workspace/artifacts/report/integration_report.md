# Integration Report — Release Gate Rename

## Summary

Renamed the release-gating approval state token from `manual_review` to `human_review_required` across backend, frontend, and operator documentation.

## Changes

### Backend (`services/api/src/`)

- `review_state.py`: `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` now use `human_review_required`. Legacy `manual_review` is normalized to `human_review_required` for compatibility.
- `routes/releases.py`: Default fallback in `build_release_gate_update` and `serialize_release_gate` is `human_review_required`. Operator hint updated to reference the new token.

### Frontend (`apps/admin/src/`)

- `components/ReleaseGateForm.tsx`: `defaultGateState.approvalState` is `human_review_required`; label updated to "Human review required".
- `lib/api.ts`: Fallback value in `buildReleaseGateRequest` and `summarizeGateEcho` is `human_review_required`.

### Documentation

- `docs/runbooks/release_gating.md`: Documents retired token (`manual_review`), active token (`human_review_required`), and the three-step verification order.

## Compatibility

- `manual_review` is still accepted by `normalize_review_state()` and mapped to `human_review_required`, satisfying the visible test `test_legacy_manual_review_normalizes`.

## Proof Artifacts

- `artifacts/dom/release_gate_capture.json` — request/echo pair showing `human_review_required` end-to-end.
