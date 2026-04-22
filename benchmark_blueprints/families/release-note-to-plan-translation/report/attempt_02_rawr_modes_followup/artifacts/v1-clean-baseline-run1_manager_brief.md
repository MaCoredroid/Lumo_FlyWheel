# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and retire the shim only after parity is proven

Start by exposing the schema drift currently hidden by the legacy parser shim so the team knows the real translator output before any downstream rollout expands blast radius.

Bounded deliverable: A verified schema-drift audit that names the translator step-id contract and confirms whether the legacy parser shim can be safely removed.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures against the audited schema

Once the translator contract is explicit, update the dependency-graph fixtures so plan-contract tests fail on future ordering regressions instead of passing under stale fixtures.

Bounded deliverable: Dependency-graph fixtures aligned to the audited step-id schema, with ordering regressions caught by contract coverage.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the dashboard summary behind the kill switch

Only after schema and fixtures agree should the translated release-plan dashboard summary be exposed in canary, and it should stay behind the kill switch because ordering failures are still weakly covered.

Bounded deliverable: A kill-switch-protected canary of the dashboard summary using the audited schema and refreshed fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update the operator runbook and launch checklist from canary-proven behavior

Document rollout and rollback steps last, after canary reveals the actual operational shape of the summary path and kill-switch handling.

Bounded deliverable: A runbook and launch checklist that reflect the proven canary and rollback procedure rather than a pre-canary guess.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill depends on knowing the real translator step-id schema; otherwise tests keep encoding the shim-masked contract.
- `RN-102` before `RN-103`: The dashboard summary must not ship until dependency-graph fixtures reflect the new schema and can catch ordering regressions.
- `RN-103` before `RN-104`: Operator guidance should be written from observed canary and rollback behavior, not from an unproven summary path.

## Primary Risk

If the dashboard summary is canaried before the translator schema audit and fixture backfill, users can see a release-plan order that looks valid in smoke runs but is actually wrong, because the dashboard renders whatever order the translator emits and current coverage does not reliably catch bad dependency ordering.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Hold dashboard exposure behind RN-101 and RN-102 so translator output and fixtures agree before users see it.
- Keep RN-103 under the kill switch during canary so incorrect ordering can be rolled back quickly.
- Delay runbook finalization until canary proves the actual rollback path.

## Assumption Ledger

- release_context_or_incident_override [missing]: No release_context/ or incident_context/ directory is present, so this plan assumes the frozen notes were not superseded by a later objective shift or rollback lesson.
- dashboard_surface_scope [observed]: The dashboard summary appears to be the first user-visible surface for translated plans, so ordering risk is prioritized there.

