# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit and resolve schema drift in the translator before removing the legacy parser shim. This de-risks downstream dashboard rendering.

Bounded deliverable: Schema drift report with resolution and legacy shim removal

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add dependency-graph fixtures so the plan contract catches ordering regressions. Required before dashboard can ship.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that validate new step-id schema

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature enabled behind kill switch, canary-only

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook and launch checklist documenting rollout and rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema must be stable before dashboard can render correctly
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema to catch ordering regressions
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary testing

## Primary Risk

Dashboard will render incorrect or malformed step ordering if schema drift is not resolved before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Audit and resolve schema drift (RN-101) before any dashboard work
- Backfill dependency-graph fixtures (RN-102) to catch ordering regressions in tests
- Enable dashboard behind kill switch (RN-103) for canary-only coverage initially

## Assumption Ledger

- Schema drift severity [missing]: Exact scope of schema drift unknown until RN-101 audit completes; planning assumes drift exists and requires resolution
- Kill switch implementation [missing]: Kill switch infrastructure details not documented; assumed to exist for dashboard summary feature
- Canary environment [missing]: Canary deployment environment configuration for RN-103 not verified; assumed available for testing

