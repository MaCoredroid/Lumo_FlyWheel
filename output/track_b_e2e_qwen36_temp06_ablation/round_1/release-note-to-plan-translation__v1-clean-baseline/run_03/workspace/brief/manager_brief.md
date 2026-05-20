# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. The shim currently masks drift in local smoke runs, so this step surfaces the real state of the schema.

Bounded deliverable: Schema drift audit report with list of drifted fields

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. The plan_contract test currently fails when new step ids are introduced without fixture updates.

Bounded deliverable: Updated fixtures that pass plan_contract tests

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable and fixtures are backfilled. Dashboard is user-visible and renders whatever the translator emits.

Bounded deliverable: Dashboard summary enabled behind kill switch in canary

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes after rollout and rollback shape are known.

Bounded deliverable: Updated runbook and launch checklist reflecting proven rollout shape

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited before backfilling fixtures, since fixtures must reflect the corrected schema.
- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema must be stable first.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema, otherwise plan_contract tests fail.
- `RN-103` before `RN-104`: Runbook only stabilizes after the dashboard summary path is proven in canary and rollout/rollback shape is known.

## Primary Risk

Dashboard renders incorrect step ordering to users if schema drift is not audited and fixtures are not backfilled before enabling the dashboard summary.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep dashboard behind kill switch until schema audit and fixture backfill are complete
- Run plan_contract tests after each step to catch ordering regressions early
- Canary-only rollout for dashboard summary before full exposure to all users

## Assumption Ledger

- Kill switch availability [to_verify]: The release notes mention a kill switch for the dashboard summary, but no implementation evidence exists in repo inventory.
- Canary deployment pipeline [missing]: No canary deployment pipeline or rollout/rollback configuration details found in repo inventory; canary process is referenced but not documented.
- Legacy parser shim removal scope [to_verify]: RN-101 mentions removing the legacy parser shim, but the scope of removal is not detailed in repo state.

