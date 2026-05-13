# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Identify schema drift masked by the legacy parser shim before removing it.

Bounded deliverable: Schema drift audit report with specific drift items documented

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add fixtures so plan contract tests catch ordering regressions when new step ids are introduced.

Bounded deliverable: Updated dependency-graph fixtures in tests/plan_contract.py that pass

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary coverage

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update runbook after the summary path is proven in canary and rollout/rollback shapes are known.

Bounded deliverable: Updated operator runbook and launch checklist reflecting proven rollout

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be identified before fixtures can be backfilled to match the corrected schema
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema and must not ship until dependency-graph fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary validation

## Primary Risk

User-visible dashboard renders incorrect step ordering if schema drift and fixture gaps are not addressed before enabling the dashboard summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Complete schema drift audit (RN-101) before any downstream changes
- Backfill dependency-graph fixtures (RN-102) and verify plan_contract.py passes
- Enable dashboard behind kill switch (RN-103) for canary-only validation before full rollout
- Hold runbook updates (RN-104) until summary path is proven in canary

## Assumption Ledger

- Schema drift extent [missing]: Full scope of schema drift masked by legacy parser shim is unknown until RN-101 audit completes
- Kill switch availability [to_verify]: Kill switch infrastructure assumed to exist for RN-103 dashboard enablement
- Canary environment [to_verify]: Canary environment assumed configured for dashboard summary testing per RN-103

