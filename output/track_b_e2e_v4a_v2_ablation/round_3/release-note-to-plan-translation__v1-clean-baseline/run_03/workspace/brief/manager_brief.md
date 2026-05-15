# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and remove legacy parser shim

Identify and document schema drift in the translator that is currently masked by the legacy parser shim. Remove the shim only after drift is understood and the schema is corrected.

Bounded deliverable: Schema drift audit report and shim removal patch

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update dependency-graph fixtures to reflect the new step-id schema. This ensures the plan contract test catches ordering regressions when new step ids are introduced.

Bounded deliverable: Updated fixtures for tests/plan_contract.py

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind kill switch

Enable the user-visible dashboard summary behind a kill switch after schema is stable and fixtures are in place. This prevents rendering incorrect order data.

Bounded deliverable: Dashboard summary feature flag enabled in canary

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update operational runbook and launch checklist with rollout and rollback procedures once the summary path is proven in canary.

Bounded deliverable: Updated runbook and launch checklist

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited and shim removed before fixtures can correctly reflect the new step-id schema
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema to prevent incorrect rendering
- `RN-103` before `RN-104`: Runbook requires known rollout and rollback shape, which is only established after dashboard summary is proven in canary

## Primary Risk

Dashboard renders incorrect step ordering to users if translator schema drift is not fixed before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Complete schema drift audit (RN-101) before any dashboard work
- Backfill fixtures (RN-102) to ensure plan contract catches ordering issues
- Use kill switch for dashboard (RN-103) to allow quick rollback if issues arise

## Assumption Ledger

- Schema drift scope [missing]: Extent and locations of schema drift not yet known until RN-101 audit completes
- Kill switch implementation [observed]: RN-103 specifies kill switch exists for dashboard summary
- Canary environment readiness [to_verify]: Canary-only coverage exists but may not be sufficient for dashboard summary validation

