# Integration Report

## Summary

Renamed the release-gating token from `manual_review` to `human_review_required`
across backend parsing, frontend submission, and operator documentation.

## Changes

- **Backend `review_state.py`**: Updated `DEFAULT_APPROVAL_STATE` and `ACTIVE_APPROVAL_STATES` to use `human_review_required`. Added `LEGACY_COMPAT_MAP` so `normalize_review_state("manual_review")` returns `human_review_required` for backward compatibility.
- **Backend `releases.py`**: Updated default fallbacks and operator_hint to reference `human_review_required`.
- **Frontend `ReleaseGateForm.tsx`**: Updated `defaultGateState.approvalState` and `operatorLabel` to use `human_review_required`.
- **Frontend `api.ts`**: Updated default fallback in `buildReleaseGateRequest` and `summarizeGateEcho` to `human_review_required`.
- **Runbook `release_gating.md`**: Documented the retired token (`manual_review`), the new token (`human_review_required`), and verification order.

## Test Alignment

- `test_legacy_manual_review_normalizes` — passes: `normalize_review_state("manual_review")` returns `"human_review_required"`.
- `test_release_gate_update_emits_new_token` — passes: `build_release_gate_update` echoes `"human_review_required"`.
- `test_release_gate_echo_uses_new_token` — passes: `serialize_release_gate` echoes `"human_review_required"`.
- `release-gate.spec.ts` — passes: request body sends `approval_state: "human_review_required"`.

## Proof Artifact

`artifacts/dom/release_gate_capture.json` contains the seeded request payload and server echo, both agreeing on `human_review_required`.
