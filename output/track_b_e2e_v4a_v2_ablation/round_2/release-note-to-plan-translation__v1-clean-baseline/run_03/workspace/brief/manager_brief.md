# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. This establishes the baseline schema understanding required for downstream work.

Bounded deliverable: Schema drift audit report identifying differences between translator output and legacy parser expectations

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. Fixtures must reflect the new schema before dashboard can consume it.

Bounded deliverable: Updated dependency-graph fixtures that pass tests/plan_contract.py with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature gated by kill switch, validated in canary environment

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook documenting rollout and rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited before fixtures can be backfilled to reflect the correct schema
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema and must not ship until dependency-graph fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook only stabilizes after rollout and rollback shape are known from canary validation

## Primary Risk

Dashboard renders incorrect or unstable step ordering if schema drift is not audited and fixtures are not updated before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Complete schema drift audit (RN-101) before any downstream work
- Backfill fixtures (RN-102) and verify plan_contract.py passes
- Keep dashboard behind kill switch (RN-103) until canary validation succeeds

## Assumption Ledger

- Schema drift scope [missing]: Exact extent of schema drift between translator and legacy parser is unknown until RN-101 audit completes
- Kill switch implementation [missing]: Kill switch infrastructure availability for RN-103 dashboard gating not confirmed
- Canary environment [missing]: Canary environment availability for RN-103 validation before RN-104 not confirmed

