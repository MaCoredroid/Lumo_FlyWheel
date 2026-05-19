# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. The shim currently masks drift in local smoke runs, so this step surfaces the real schema state.

Bounded deliverable: Schema drift audit report with list of drifted fields and shim removal plan.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. Tests currently fail when new step ids are introduced without fixture updates.

Bounded deliverable: Updated fixtures that pass plan_contract.py with the new schema step ids.

Evidence:
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable and fixtures are backfilled. The dashboard is user-visible.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary coverage.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes after rollout and rollback shape are known.

Bounded deliverable: Updated runbook and launch checklist reflecting proven canary rollout and rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the audited schema; backfilling before the audit would use stale schema definitions.
- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema must be stable first.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema, otherwise plan_contract.py will fail on ordering regressions.
- `RN-103` before `RN-104`: Runbook and launch checklist require the dashboard summary path to be proven in canary so rollout and rollback shapes are known.

## Primary Risk

Dashboard renders incorrect step ordering to users if shipped before schema audit and fixture backfill are complete.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep dashboard behind a kill switch (RN-103) until schema and fixtures are verified.
- Run plan_contract.py after fixture backfill to confirm ordering regressions are caught.
- Gate runbook update (RN-104) behind successful canary proof of the summary path.

## Assumption Ledger

- Legacy parser shim removal timeline [missing]: RN-101 audits drift before shim removal, but the actual shim removal date and rollback plan are not specified in any release note or repo inventory.
- Canary rollout criteria for dashboard [missing]: No explicit acceptance criteria defined for when the dashboard summary is considered proven in canary.
- Schema drift severity [missing]: Extent of masked schema drift is unknown until RN-101 audit completes; if drift is breaking, RN-101 scope expands.

