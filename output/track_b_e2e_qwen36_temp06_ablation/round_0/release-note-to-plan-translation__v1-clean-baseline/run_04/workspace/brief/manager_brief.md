# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator schema drift before removing the legacy parser shim. The shim currently masks drift in local smoke runs, so this audit de-risks all downstream work.

Bounded deliverable: Schema drift report with identified mismatches between translator output and legacy shim expectations.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions. Fixtures must reflect the new schema discovered in RN-101.

Bounded deliverable: Updated fixture files that pass plan_contract.py with the new step-id schema.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch. Only after schema is stable (RN-101) and fixtures are backfilled (RN-102).

Bounded deliverable: Dashboard summary feature flag behind kill switch, validated in canary.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. Runbook only stabilizes after rollout and rollback shape are known.

Bounded deliverable: Updated runbook and launch checklist documenting rollout and rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the new schema discovered during the schema drift audit.
- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema produced by the translator; schema must be stable first.
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after the rollout and rollback shape are known from canary validation.

## Primary Risk

The release dashboard renders incorrect step ordering to operators if the dashboard summary (RN-103) ships before the schema audit (RN-101) and fixture backfill (RN-102) are complete.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep the dashboard summary behind a kill switch (RN-103) so it can be disabled immediately if bad data appears.
- Complete schema audit (RN-101) and fixture backfill (RN-102) before enabling the dashboard, ensuring the translator output is correct.
- Validate in canary before broad rollout; canary coverage exists for dashboard summary (test_inventory.md).

## Assumption Ledger

- Kill switch implementation [missing]: No evidence of an existing kill switch mechanism for the dashboard summary; RN-103 assumes one exists or can be added.
- Canary environment readiness [missing]: Canary coverage exists but its readiness for validating the full dashboard summary path is unconfirmed.

