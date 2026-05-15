# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Identify and document schema drift between translator output and legacy parser shim expectations before removing the shim.

Bounded deliverable: Schema drift audit report with concrete field mappings and impact assessment.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add test fixtures reflecting the new schema so plan contract tests catch ordering regressions.

Bounded deliverable: Updated test fixtures in tests/plan_contract.py that pass with new step ids.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable.

Bounded deliverable: Dashboard summary feature toggled on for canary with kill switch for rollback.

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Document rollout and rollback procedures once the dashboard summary is proven in canary.

Bounded deliverable: Updated runbook and checklist reflecting actual rollout shape.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema audit must complete before fixtures can be written to match the correct schema.
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary.

## Primary Risk

Dashboard renders incorrect or unstable step ordering if schema drift is not resolved before enabling the summary.

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Complete schema audit (RN-101) before any dashboard work.
- Add dependency-graph fixtures (RN-102) to catch ordering regressions in tests.
- Use kill switch (RN-103) for immediate rollback if dashboard issues appear.

## Assumption Ledger

- Legacy parser shim removal [missing]: Release notes mention removing the shim but no explicit task or timeline is provided; assuming removal happens after RN-101 audit.
- Canary rollout window [missing]: No duration specified for canary validation period before RN-104 runbook update.

