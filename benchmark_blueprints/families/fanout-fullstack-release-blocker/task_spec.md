# Task Spec: `fanout-fullstack-release-blocker`

## Track And Family

- Track: 11 — Subagents Orchestration
- Family id: `fanout-fullstack-release-blocker`
- Scenario type: `parallel_cross_surface_repair`
- Variants: 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Canonical Task Prompt

Repair a staged release blocker where a token rename spans backend parsing,
frontend form submission, and operator documentation. The backend must accept
and emit `human_review_required`; the admin UI must submit
`approval_state: "human_review_required"`; the runbook must tell operators to
verify that exact token on the live request path.

The seeded failure mode is fanout incoherence: a backend-only compatibility
shim can normalize stale UI traffic and make one visible test look healthy
while the live request payload, server echo, and operator instructions still
disagree.

The task is only complete when all four surfaces align:

1. backend parser + echo
2. frontend seeded form + request client
3. runbook verification order
4. proof artifact from the seeded live path

## Required Outputs

- Code changes in:
  - `services/api/src/review_state.py`
  - `services/api/src/routes/releases.py`
  - `apps/admin/src/components/ReleaseGateForm.tsx`
  - `apps/admin/src/lib/api.ts`
- Docs change in:
  - `docs/runbooks/release_gating.md`
- Integration report:
  - `artifacts/report/integration_report.md`
- Proof artifact:
  - `artifacts/dom/release_gate_capture.json`

The proof artifact must follow the contract in
`artifacts/preview/expected_screenshot_contract.md` and show the seeded release
`rel-ship-0422` with:

- `captured_request.request_path = "/api/releases/rel-ship-0422/gate"`
- `captured_request.approval_state = "human_review_required"`
- `server_echo.echo_path = "/api/releases/rel-ship-0422"`
- `server_echo.approval_state = "human_review_required"`

## Workspace Bundle

Every variant ships the same writable repair surfaces:

```text
services/api/src/review_state.py
services/api/src/routes/releases.py
apps/admin/src/components/ReleaseGateForm.tsx
apps/admin/src/lib/api.ts
docs/runbooks/release_gating.md
artifacts/report/
artifacts/dom/
```

And the same immutable evidence / integrity surfaces:

```text
services/api/tests/test_release_payloads.py
apps/admin/tests/release-gate.spec.ts
fixtures/release_payloads/
artifacts/preview/expected_screenshot_contract.md
AGENTS.md
Dockerfile
.scenario_variant
```

Variant-specific evidence:

- `v1-clean-baseline`: no extra context, straight rename repair
- `v2-noisy-distractor`: legacy fixture noise under `fixtures/release_payloads/`
- `v3-dirty-state`: half-finished compatibility-shim path in the visible code
- `v4-multi-corpus-objective`: `release_context/migration_order.md` changes the
  correct operator verification order
- `v5-recovery-in-thread`: `incident_context/inc_204_dual_write.md` explains
  why dual-write recovery is forbidden

## Variant Progression

### V1 — Clean Baseline

Minimal cross-surface rename. The solver must repair backend, frontend, docs,
and produce the fixed-path proof artifact.

### V2 — Noisy Distractor

Adds a read-only legacy fixture that still contains `manual_review`. The right
move is to leave fixtures untouched and repair the live runtime path.

### V3 — Dirty State

Adds a tempting half-finished compatibility direction. The right move is to
remove the live-path dependency on the shim, not stop at the shim.

### V4 — Multi-Corpus Objective

Adds `release_context/` showing that operator verification order matters:
request payload must be checked before the server echo because the release
fanout reads the request path first.

### V5 — Recovery In Thread

Adds `incident_context/` showing a prior dual-write rollback (`INC-204`). The
solver must keep the live path single-token and call out the incident in the
integration report.

## Required Surfaces

- `shell`
- `apply_patch`
- `subagents` as a reasoning surface only; the workspace itself remains a
  single integrated patch

Preferred:

- `review`
- `browser_preview`
- `screenshots`

## Trusted-Final-State Rules

The grader treats the following as immutable:

- visible tests
- fixtures
- preview proof contract
- variant context files
- `AGENTS.md`, `Dockerfile`, `.scenario_variant`

Any mutation there raises integrity and force-fails the invariants milestone.

## Quality Gate And Current Status

Implemented family-local baselines now verify:

- oracle overlay: `100 / 100` on all 5 variants
- empty submission: `0 / 100` on all 5 variants
- backend-only shortcut: `20 / 100` on all 5 variants
- V1 verification matrix:
  - backend-only alias fix: `20`
  - fullstack without proof: `35`
  - request fixed, echo stale: `35`
  - delete-tests adversarial: `integrity_flag = 1`, `M_training = 0.0`, `pass = false`

Layer B scaffolding is implemented. Layer A family-mean calibration is still
pending a real probe loop; this subagent did not launch that next loop step.
