# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Audit the translator schema drift and remove the legacy parser shim that currently masks schema drift in local smoke runs.

Bounded deliverable: Schema audit report and legacy shim removal patch.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions with the new step ids.

Bounded deliverable: Updated fixtures and passing plan_contract tests.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind kill switch

Enable the dashboard summary behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary coverage.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook and launch checklist.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema must be audited and stable first.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema so plan contract catches ordering regressions.
- `RN-103` before `RN-104`: Runbook and launch checklist only stabilize after the summary path is proven in canary and rollout/rollback shape is known.

## Primary Risk

If the dashboard summary ships before schema audit and fixture backfill, it will render incorrect or stale step ordering to users because the legacy parser shim masks schema drift.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep dashboard behind kill switch until schema audit (RN-101) and fixture backfill (RN-102) are complete.
- Run canary-only coverage on dashboard summary before full rollout.
- Monitor plan_contract test results to catch ordering regressions early.

## Assumption Ledger

- Kill switch implementation exists and is tested. [missing]: No evidence in repo inventory that a kill switch mechanism for the dashboard summary already exists or has been tested.
- Canary environment availability [missing]: Release notes assume a canary environment exists for proving the summary path; no repo evidence confirms this.

