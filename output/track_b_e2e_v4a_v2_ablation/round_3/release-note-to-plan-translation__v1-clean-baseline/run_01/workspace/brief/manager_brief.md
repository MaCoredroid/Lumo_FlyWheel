# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and remove legacy parser shim

Identify and resolve schema drift in the translator before any downstream work, removing the legacy parser shim that masks drift in local smoke runs.

Bounded deliverable: Schema drift report with concrete fixes applied; legacy shim removed and smoke tests passing.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures for plan contract

Add fixtures reflecting the new step-id schema so the plan contract catches ordering regressions.

Bounded deliverable: tests/plan_contract.py passing with new step ids; fixture coverage for dependency ordering.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard behind kill switch

Enable the dashboard summary path behind a kill switch once schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary enabled in canary with kill switch for rollback.

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Stabilize runbook and checklist after rollout and rollback procedures are proven in canary.

Bounded deliverable: Updated runbook with verified rollout/rollback procedures; launch checklist complete.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be resolved before fixtures can accurately reflect the new step-id schema.
- `RN-102` before `RN-103`: Dashboard summary consumes the step-id schema and must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary validation.

## Primary Risk

Dashboard renders incorrect step ordering to users if translator schema drift is not fixed before enabling the dashboard summary.

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Complete schema drift audit and shim removal before enabling dashboard (RN-101 first).
- Use kill switch for dashboard summary to allow immediate rollback if ordering issues appear.
- Ensure plan contract tests pass with new fixtures before canary enable.

## Assumption Ledger

- Schema drift scope [missing]: Exact extent of schema drift is unknown until RN-101 audit is complete.

