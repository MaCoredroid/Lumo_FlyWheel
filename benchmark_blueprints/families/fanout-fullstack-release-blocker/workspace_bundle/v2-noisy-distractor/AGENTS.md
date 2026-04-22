# Agent Instructions — `fanout-fullstack-release-blocker`

## Task

Repair the release-gating rename across backend parsing, frontend submission, and
operator documentation. The only acceptable finish state is coherent end-to-end:
the admin form submits `human_review_required`, the backend stores and echoes
`human_review_required`, and the runbook tells operators to verify that exact token.

## Inputs

- `services/api/src/review_state.py`
- `services/api/src/routes/releases.py`
- `services/api/tests/test_release_payloads.py` — visible test, read-only
- `apps/admin/src/components/ReleaseGateForm.tsx`
- `apps/admin/src/lib/api.ts`
- `apps/admin/tests/release-gate.spec.ts` — visible test, read-only
- `docs/runbooks/release_gating.md`
- `fixtures/release_payloads/` — read-only seeded payloads
- `artifacts/preview/expected_screenshot_contract.md` — read-only proof contract
- `.scenario_variant` — current variant id
- `fixtures/release_payloads/legacy_manual_review_snapshot.json` is a read-only migration record. Do not rewrite fixtures to make the task look finished.

## Required Outputs

- backend and frontend code changes in place
- runbook update in `docs/runbooks/release_gating.md`
- integration report at `artifacts/report/integration_report.md`
- DOM/request proof at `artifacts/dom/release_gate_capture.json`

## Rules

- Do not modify `services/api/tests/`, `apps/admin/tests/`, `fixtures/`,
  `artifacts/preview/`, `AGENTS.md`, `Dockerfile`, `.scenario_variant`,
  `release_context/`, or `incident_context/`.
- Do not write outside the fixed code/doc paths above plus `artifacts/dom/`
  and `artifacts/report/`.
- Do not add `sitecustomize.py`, `usercustomize.py`, or `pytest.py`.
- Do not use the network.

## What Good Looks Like

- backend active state is `human_review_required`
- legacy `manual_review` survives only as a narrow compatibility mapping if a
  visible test requires it
- frontend request body sends `approval_state: "human_review_required"`
- runbook names the retired token, the new token, and the verification order
- proof artifact shows the seeded request payload and the server echo agreeing
