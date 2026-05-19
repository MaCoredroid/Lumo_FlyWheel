# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit the translator for schema drift before removing the legacy parser shim. The shim currently masks drift in local smoke runs, so this step surfaces the real state of the step-id schema.

Bounded deliverable: Schema drift audit report with a decision on shim removal readiness.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract test catches ordering regressions. Required before the dashboard can ship because the contract test fails when new step ids appear without fixture updates.

Bounded deliverable: Updated fixtures that pass tests/plan_contract.py with new step ids.

Evidence:
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch once the schema is stable and fixtures are in place. The dashboard is user-visible and will render whatever order the translator emits.

Bounded deliverable: Dashboard summary feature flag gated behind kill switch, canary-validated.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes after rollout and rollback shape are known from the dashboard canary.

Bounded deliverable: Updated runbook and launch checklist reflecting proven rollout/rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the audited schema; backfilling before the audit risks encoding stale schema drift.
- `RN-101` before `RN-103`: Dashboard consumes the step-id schema produced by the translator; schema must be stable first.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema per dependency_map.md.
- `RN-103` before `RN-104`: Runbook and launch checklist depend on the summary path being proven in canary.

## Primary Risk

If the dashboard summary (RN-103) ships before schema audit and fixture backfill are complete, the user-visible release dashboard will render incorrect step ordering, misleading operators during rollout.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep the dashboard behind a kill switch (RN-103) so it can be disabled instantly if bad ordering appears.
- Complete the schema audit (RN-101) before any dashboard work to surface hidden drift masked by the shim.
- Backfill fixtures (RN-102) and verify plan_contract.py passes before enabling the dashboard in canary.
- Limit initial dashboard exposure to canary only, not full rollout.

## Assumption Ledger

- Legacy parser shim removal timeline [missing]: The release notes say to audit before removing the shim, but no timeline or owner for shim removal is documented. The audit report (RN-101 deliverable) must include a shim removal decision.
- Canary environment readiness for dashboard [missing]: Assuming the canary environment has the dependency-graph fixtures and translator schema available; test_inventory.md only mentions canary coverage exists, not that the environment is configured for this release.
- Rollback procedure for dashboard kill switch [missing]: Assuming the kill switch can be toggled without a deploy; no evidence in repo_inventory confirms this operational detail.

