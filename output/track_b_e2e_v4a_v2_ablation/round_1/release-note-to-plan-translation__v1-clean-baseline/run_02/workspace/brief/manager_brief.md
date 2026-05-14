# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Identify and document all schema drift between translator output and expected step-id schema. The legacy parser shim currently masks this drift in local smoke runs.

Bounded deliverable: Schema drift audit report with list of discrepancies and required translator fixes

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures for plan contract

Add dependency-graph fixtures so the plan contract test catches ordering regressions when new step ids are introduced.

Bounded deliverable: Updated test fixtures in tests/plan_contract.py that pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind kill switch

Enable the user-visible dashboard summary feature behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary enabled in canary with kill switch, verified against new schema

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Document rollout and rollback procedures once the summary path is proven in canary.

Bounded deliverable: Updated runbook with verified rollout/rollback procedures and launch checklist

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited and fixed before fixtures can accurately reflect the new schema
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema per repo_inventory/dependency_map.md
- `RN-103` before `RN-104`: Runbook only stabilizes after rollout and rollback shape are known from canary validation

## Primary Risk

Dashboard renders incorrect step ordering to users if translator schema drift is not fixed before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Complete RN-101 schema audit before any dashboard work
- Enable dashboard behind kill switch (RN-103) to allow quick rollback
- Backfill dependency-graph fixtures (RN-102) so plan contract catches ordering regressions

## Assumption Ledger

- release_context and incident_context directories [missing]: Optional context directories do not exist in this workspace; proceeding with frozen release notes and repo inventory only
- Kill switch implementation details [missing]: Kill switch infrastructure exists for RN-103; no additional work needed to add feature flag capability

