# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. The legacy shim currently masks schema drift in local smoke runs, so this audit must happen first to understand the full scope of schema issues.

Bounded deliverable: Schema drift audit report documenting all translator schema inconsistencies

Evidence:
- `repo_inventory/repo_state.md`
- `release_notes/release_notes_2026_04.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. The test plan_contract.py currently fails when new step ids are introduced without fixture updates.

Bounded deliverable: Updated dependency-graph fixtures that pass plan_contract.py tests

Evidence:
- `repo_inventory/test_inventory.md`
- `release_notes/release_notes_2026_04.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable. The dashboard consumes the step-id schema produced by the translator and must not ship until fixtures reflect the new schema.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary validation

Evidence:
- `repo_inventory/dependency_map.md`
- `release_notes/release_notes_2026_04.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes after the rollout and rollback shape are known.

Bounded deliverable: Updated operator runbook and launch checklist

Evidence:
- `repo_inventory/dependency_map.md`
- `release_notes/release_notes_2026_04.md`

## Dependency Notes

- `RN-101` before `RN-103`: The translator schema audit must complete before the dashboard can be enabled, as the dashboard consumes the step-id schema
- `RN-102` before `RN-103`: Dependency-graph fixtures must reflect the new schema before the dashboard summary can ship, otherwise the plan contract will fail on ordering regressions
- `RN-103` before `RN-104`: The runbook only stabilizes after the rollout and rollback shape are known from canary validation of the dashboard summary

## Primary Risk

The release dashboard will render incorrect step ordering to users if the schema and fixtures are not aligned before enabling the dashboard summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Enable dashboard summary behind a kill switch for controlled rollout
- Validate schema stability before removing legacy parser shim
- Use canary-only coverage to verify dashboard summary behavior before full rollout

## Assumption Ledger

- Legacy parser shim removal timeline [missing]: No explicit timeline provided for when the legacy parser shim will be removed after schema audit
- Canary deployment criteria [to_verify]: Specific success criteria for canary validation of dashboard summary are not documented
- Schema drift scope [observed]: Schema drift is masked by legacy shim in local smoke runs as documented in repo_state.md

