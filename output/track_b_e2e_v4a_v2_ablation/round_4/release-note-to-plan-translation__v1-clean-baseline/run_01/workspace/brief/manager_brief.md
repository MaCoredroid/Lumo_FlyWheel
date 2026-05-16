# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit and document schema drift in the translator before removing the legacy parser shim. This exposes the true schema state that the shim currently masks.

Bounded deliverable: Schema drift audit report with documented differences between expected and actual step-id schema

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add dependency-graph fixtures to the plan contract so ordering regressions are caught by tests. Required before dashboard can safely ship.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that validate step-id ordering

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch now that schema is stable and fixtures validate ordering.

Bounded deliverable: Dashboard summary feature flag enabled in canary with kill switch available

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update operator runbook and launch checklist once the summary path is proven stable in canary rollout.

Bounded deliverable: Updated runbook documenting rollout and rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited and resolved before fixtures can be correctly backfilled; the legacy shim masks the true schema state
- `RN-102` before `RN-103`: Dashboard summary consumes the step-id schema and must not ship until dependency-graph fixtures reflect the new schema to prevent ordering regressions
- `RN-103` before `RN-104`: Runbook and launch checklist depend on knowing the actual rollout and rollback shape, which is only determined after canary proves the summary path

## Primary Risk

Dashboard renders incorrect step ordering to users if enabled before schema drift is audited and fixtures are backfilled

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep dashboard behind kill switch until RN-101 and RN-102 are complete
- Add fixture-based contract tests to catch ordering regressions before canary
- Validate schema stability in canary before broader rollout

## Assumption Ledger

- schema_drift_magnitude [missing]: Extent of schema drift masked by legacy parser shim is unknown until RN-101 audit completes

