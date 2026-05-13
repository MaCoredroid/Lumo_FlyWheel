# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit and resolve schema drift in the translator before removing the legacy parser shim. This de-risks subsequent work from masking issues.

Bounded deliverable: Schema drift audit report with identified gaps and a plan to remove the legacy parser shim.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add dependency-graph fixtures so the plan contract catches ordering regressions when new step ids are introduced.

Bounded deliverable: Updated test fixtures in tests/plan_contract.py that validate dependency ordering.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature gated behind a kill switch, ready for canary rollout.

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook documenting rollout and rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema drift must be resolved first.
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema to catch ordering regressions.
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary testing.

## Primary Risk

The release dashboard will render incorrect or unstable ordering if the translator schema drift is not resolved before enabling the dashboard summary.

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Audit schema drift (RN-101) before any dashboard work
- Backfill dependency-graph fixtures (RN-102) to catch ordering regressions in tests
- Keep dashboard summary behind a kill switch during canary validation

## Assumption Ledger

- legacy parser shim removal timeline [missing]: No explicit timeline for shim removal provided in release notes; assumed to be addressed after schema audit completes.
- Canary environment readiness [missing]: No explicit confirmation that canary environment is configured for dashboard summary testing.

