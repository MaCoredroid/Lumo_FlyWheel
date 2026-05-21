# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and remove legacy parser shim

Audit and fix the translator schema drift that the legacy parser shim currently masks in local smoke runs. Removing the shim exposes the real schema state so downstream work builds on correct data.

Bounded deliverable: Schema drift audit report with shim removal, passing local smoke runs against the new schema.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill the dependency-graph fixtures so the plan contract test catches ordering regressions with the new step ids from the corrected schema.

Bounded deliverable: Updated fixtures that make tests/plan_contract.py pass with new step ids.

Evidence:
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch now that the schema is stable and fixtures are in place. Canary coverage already exists for the dashboard.

Bounded deliverable: Dashboard summary feature flag behind kill switch, canary-validated.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist now that the summary path is proven in canary and the rollout/rollback shape is known.

Bounded deliverable: Updated runbook and launch checklist reflecting proven rollout and rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the corrected schema; backfilling before the schema audit would use stale step ids.
- `RN-101` before `RN-103`: Dashboard consumes the translator step-id schema; enabling it before schema stabilization risks rendering incorrect plan order to users.
- `RN-102` before `RN-103`: Dependency map explicitly requires fixtures to reflect the new schema before the dashboard ships; otherwise plan_contract.py will fail on new step ids.
- `RN-103` before `RN-104`: Runbook needs the proven rollout and rollback shape from the dashboard canary before it can be finalized.

## Primary Risk

Dashboard renders incorrect release-plan ordering to users if enabled before schema audit and fixture backfill are complete.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep dashboard behind a kill switch (RN-103) so it can be disabled instantly if regressions appear.
- Complete schema audit (RN-101) and fixture backfill (RN-102) before enabling the dashboard.
- Rely on existing canary coverage for dashboard summary to catch regressions early.

## Assumption Ledger

- Legacy parser shim removal [missing]: No evidence of other consumers depending on the shim; assuming shim removal is safe once schema drift is audited and fixed.
- Kill switch implementation [missing]: No repo evidence about whether a kill-switch mechanism already exists for the dashboard; assuming one can be added or the existing feature flag is sufficient.
- Canary environment readiness [missing]: test_inventory.md mentions canary-only coverage but provides no deployment details; assuming canary environment is available and configured.

