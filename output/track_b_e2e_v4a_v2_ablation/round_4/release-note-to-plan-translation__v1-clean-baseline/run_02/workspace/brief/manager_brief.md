# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Identify and document schema drift in the translator before removing the legacy parser shim that currently masks the issue.

Bounded deliverable: Schema drift audit report with list of divergent fields and affected paths

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update test fixtures to reflect the new schema so the plan contract catches ordering regressions.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Activate the translated release-plan dashboard summary feature with a kill switch once schema is stable.

Bounded deliverable: Dashboard summary feature enabled behind kill switch in canary

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Document rollout and rollback procedures after the summary path is proven in canary.

Bounded deliverable: Updated runbook with rollout/rollback procedures and launch checklist

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited and understood before fixtures can be correctly backfilled to reflect the new schema
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema from translator and must not ship until dependency-graph fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook requires known rollout and rollback shapes which are only established after dashboard summary is proven in canary

## Primary Risk

Dashboard renders incorrect step ordering to users if schema drift is not audited and fixtures are not backfilled before enabling the dashboard

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Complete schema drift audit (RN-101) before any dashboard work
- Backfill fixtures and verify plan_contract.py passes (RN-102) before enabling dashboard
- Keep dashboard behind kill switch for rapid rollback if ordering issues detected

## Assumption Ledger

- schema_drift_severity [missing]: No detailed inventory of schema drift exists; audit in RN-101 will reveal actual scope
- kill_switch_availability [missing]: Kill switch infrastructure for dashboard summary feature not explicitly documented in repo_inventory
- canary_rollback_procedures [missing]: Rollback procedures for dashboard feature not documented until RN-104 completes

