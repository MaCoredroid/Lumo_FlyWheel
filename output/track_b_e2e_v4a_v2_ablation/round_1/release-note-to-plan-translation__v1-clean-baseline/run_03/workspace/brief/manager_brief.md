# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. This bounded work de-risks subsequent steps by exposing hidden schema issues.

Bounded deliverable: Schema drift audit report with concrete findings on translator output vs expected step-id schema

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. Required after schema audit to ensure fixtures reflect the actual schema.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature behind kill switch, canary-validated

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Operator runbook and launch checklist updated with proven rollout/rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema audit must complete before fixtures can be correctly backfilled to reflect actual translator output
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema from translator and must not ship until dependency-graph fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary validation

## Primary Risk

Dashboard renders incorrect or unstable step ordering to users if translator schema drift is not audited and fixtures are not updated before enabling the feature

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Complete schema audit (RN-101) before any downstream work to expose hidden drift
- Backfill fixtures (RN-102) and verify tests/plan_contract.py passes before enabling dashboard
- Keep dashboard behind kill switch (RN-103) during canary to limit blast radius
- Validate rollout/rollback procedures in canary before updating runbook (RN-104)

## Assumption Ledger

- Schema drift severity [missing]: Audit may reveal minimal drift, but proceeding without audit risks user-visible failures
- Canary coverage for bad ordering [missing]: Test inventory states no coverage exists for bad dependency ordering; mitigation relies on kill switch
- Kill switch availability [to_verify]: Kill switch infrastructure assumed to be in place for dashboard summary

