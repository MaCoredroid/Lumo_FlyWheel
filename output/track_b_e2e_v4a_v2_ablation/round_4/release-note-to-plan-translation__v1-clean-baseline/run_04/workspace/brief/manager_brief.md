# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. This de-risks subsequent work by exposing any hidden coupling between the legacy shim and current schema.

Bounded deliverable: Schema drift audit report with explicit findings on shim dependencies

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. This enables test coverage for the new schema before the dashboard consumes it.

Bounded deliverable: Updated fixtures that make tests/plan_contract.py pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary

Enable the dashboard summary behind a kill switch after the schema is stable. The dashboard consumes the step-id schema and must not ship until fixtures reflect the new schema.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary coverage

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes after the rollout and rollback shape are known.

Bounded deliverable: Updated runbook and launch checklist with proven rollout/rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited before fixtures can be correctly backfilled; the legacy shim masks drift that would cause incorrect fixture data
- `RN-102` before `RN-103`: Dashboard summary consumes the step-id schema and must not ship until dependency-graph fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary testing of the dashboard summary

## Primary Risk

Dashboard renders incorrect step ordering to users if translator schema drift is not audited and fixtures are not updated before enabling the dashboard

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Audit schema drift (RN-101) before any dashboard work to expose hidden coupling
- Backfill fixtures (RN-102) and verify tests/plan_contract.py passes before enabling dashboard
- Enable dashboard behind a kill switch (RN-103) with canary-only coverage first
- Complete runbook updates (RN-104) only after canary proves the rollout shape

## Assumption Ledger

- Schema drift extent [missing]: The actual extent of schema drift masked by the legacy parser shim is unknown until RN-101 audit is complete
- Fixture gap [missing]: Assuming dependency-graph fixtures can be backfilled without breaking existing plan contract tests
- Kill switch availability [missing]: Assuming kill switch infrastructure exists for the dashboard summary feature

