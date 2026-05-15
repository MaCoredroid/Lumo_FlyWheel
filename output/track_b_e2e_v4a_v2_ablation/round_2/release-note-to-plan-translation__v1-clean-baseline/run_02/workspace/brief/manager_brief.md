# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit and document schema drift in the translator before removing the legacy parser shim. This establishes the baseline for what the new schema must produce.

Bounded deliverable: Schema drift audit report with concrete differences between current translator output and expected step-id schema

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update dependency-graph fixtures to reflect the new schema so the plan contract catches ordering regressions.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema is stable and fixtures are backfilled.

Bounded deliverable: Dashboard summary feature behind kill switch, canary-only

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated operator runbook with rollout/rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift audit must complete before fixtures can be backfilled to the correct schema
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema from translator and must not ship until dependency-graph fixtures reflect new schema
- `RN-103` before `RN-104`: Runbook only stabilizes after rollout and rollback shape are known from canary

## Primary Risk

Dashboard renders incorrect or unstable step ordering to users if schema drift is not audited and fixtures are not backfilled before enabling the dashboard summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Complete schema drift audit (RN-101) before any dashboard work
- Backfill dependency-graph fixtures (RN-102) before enabling dashboard
- Keep dashboard behind kill switch for canary-only exposure until proven

## Assumption Ledger

- legacy parser shim removal is deferred until after schema audit completes [observed]: RN-101 says audit before removing shim; plan assumes shim stays until audit is done
- release_context and incident_context files [missing]: Optional context directories not present in workspace; proceeding with frozen release notes and repo inventory only
- canary deployment coverage [observed]: test_inventory.md notes canary-only coverage exists for dashboard summary but not for bad dependency ordering; assuming this gap is acceptable for initial launch

