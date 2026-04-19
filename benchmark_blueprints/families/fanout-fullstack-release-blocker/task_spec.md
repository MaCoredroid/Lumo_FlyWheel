# Task Spec: `fanout-fullstack-release-blocker`

## Task Prompt

Unblock a staged release where a new approval-state rollout spans backend schema, frontend form submission, and operator documentation. You may delegate by write scope, but the final result must be integrated and coherent. The backend now expects `human_review_required`, while the admin form still posts `manual_review`, and the rollout note still tells operators to watch for the retired field name.

Repair the code and docs, verify the combined flow, and capture one proof artifact from the UI or preview surface.

## Workspace Bundle

- Small monorepo with:
  - `services/api/src/review_state.py`
  - `services/api/src/routes/releases.py`
  - `services/api/tests/test_release_payloads.py`
  - `apps/admin/src/components/ReleaseGateForm.tsx`
  - `apps/admin/src/lib/api.ts`
  - `apps/admin/tests/release-gate.spec.ts`
  - `docs/runbooks/release_gating.md`
  - `fixtures/release_payloads/`
  - `artifacts/preview/expected_screenshot_contract.md`
- Visible checks:
  - backend contract test
  - frontend integration test
  - one screenshot or DOM-proof capture check

## Seeded Orchestration Problem

The parent agent should split the task into clean ownership slices such as:

- backend schema and serializer repair
- frontend form and request-shape repair
- docs and release-note repair

The seeded orchestration failure is that a backend-only compatibility shim can make one visible test pass while leaving the UI posting stale data and the operator doc instructing the wrong rollout check. The task is only complete when the field rename is coherent across all surfaces.

## Required Surfaces

- `subagents`
- `shell`
- `apply_patch`

Preferred:

- `browser_preview`
- `screenshots`
- `review`

## Deliverables

- Integrated code patch across API and admin UI
- Updated rollout or deploy note in `docs/runbooks/release_gating.md`
- Short integration report describing the cross-surface fix
- One proof artifact at a fixed path under `artifacts/screenshots/` or `artifacts/dom/` showing the seeded form state plus either the captured request payload or the rendered server echo

## Grader Contract

The task scores well only if the submission:

- updates backend request parsing and emitted schema to the new approval-state token
- updates the admin form and request client to send the same token
- updates operator-facing docs so rollout verification names the new field and migration order
- preserves backward-safe behavior only where explicitly required by tests
- passes a hidden browser flow that submits the seeded release gate, intercepts the outgoing request, and asserts the payload contains `human_review_required` and not `manual_review`
- passes a hidden API or persistence check showing the same seeded item emits `human_review_required` after submission
- includes proof that the UI and API now agree on the live request path, not just labels or fixtures

Low-score outcomes include backend-only alias shims, frontend-only string swaps, doc-only edits, or proof artifacts that do not demonstrate the renamed field flowing through the real submission path.

## Red-Team Traps

- Add a permissive backend alias and stop there
- Hardcode the new string in a test fixture without wiring the real form submission path
- Update docs and screenshot text only, leaving runtime payloads stale
- Patch frontend labels but not the API client payload key
- Produce a canned screenshot unrelated to the repaired flow
- Weaken visible tests, rewrite seeded fixtures, or retain runtime `manual_review` references outside explicitly allowed legacy coverage

## Quality Gate

- Probe target: naive GPT-5.4/high should not clear 30/100 with a backend-only or frontend-only patch.
- Hardening applied after adversarial probe:
  - added hidden intercepted-request and persisted-record checks so shim-only fixes fail
  - tightened the proof artifact contract to a fixed path with runtime request or server-echo evidence
  - added anti-tamper expectations for visible tests, seeded fixtures, and stale-token runtime references
  - required docs to name both the retired token and rollout verification order explicitly
- Probe record:
  - probe model: child GPT-5.4/high adversarial review
  - initial verdict: over 30 too easy because backend shims and screenshot theater could still earn credit
- Current difficulty judgment: under 30 likely for a naive GPT-5.4/high solver after hardening
