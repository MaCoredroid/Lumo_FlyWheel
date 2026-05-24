# Integration report

Summary

The release-gating rename from `manual_review` to `human_review_required` has been applied across all code paths.

## Changes

- Backend `review_state.py`: DEFAULT_APPROVAL_STATE and ACTIVE_APPROVAL_STATES use `human_review_required`. Legacy `manual_review` is mapped via LEGACY_STATE_MAP for compatibility.
- Backend `releases.py`: default values and operator_hint updated to `human_review_required`.
- Frontend `ReleaseGateForm.tsx`: defaultGateState.approvalState is `human_review_required`.
- Frontend `api.ts`: buildReleaseGateRequest and summarizeGateEcho use `human_review_required`.
- Runbook `release_gating.md`: references `human_review_required` throughout.

## Verification

- `normalize_review_state("manual_review")` -> "human_review_required"
- `build_release_gate_update` emits `human_review_required`
- `serialize_release_gate` echoes `human_review_required`
- Frontend form submits `human_review_required`
- Runbook instructs operators to verify `human_review_required`

## Proof Artifact

- `artifacts/dom/release_gate_capture.json`: request and server echo agree on `human_review_required`.
