# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim to identify all schema inconsistencies that are currently masked.

Bounded deliverable: Schema drift audit report documenting all discrepancies between translator output and expected step-id schema.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update dependency-graph fixtures to reflect the new schema so the plan contract can catch ordering regressions.

Bounded deliverable: Updated test fixtures in tests/plan_contract.py that validate the new step-id schema.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature enabled behind kill switch with canary-only coverage.

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook documenting rollout and rollback procedures for the dashboard summary.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited before fixtures can be correctly backfilled to reflect the new schema.
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema and must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary deployment.

## Primary Risk

Dashboard renders incorrect or broken step ordering to users if enabled before schema is stable and fixtures are updated.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Enable dashboard behind kill switch (RN-103) only after schema audit (RN-101) and fixture backfill (RN-102) are complete
- Validate canary deployment before broader rollout
- Ensure plan_contract.py tests pass with updated fixtures before enabling dashboard

## Assumption Ledger

- schema_drift_scope [missing]: The schema drift audit may reveal more extensive translator changes than anticipated; scope to be determined during RN-101.
- kill_switch_availability [missing]: Kill switch infrastructure for dashboard summary feature flag needs verification.
- canary_environment [missing]: Canary environment availability and configuration for RN-103 validation needs confirmation.

