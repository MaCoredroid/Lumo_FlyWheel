# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Identify all schema drift masked by the legacy parser shim so the translator emits correct step-id output. This is the foundation for downstream dashboard and fixture work.

Bounded deliverable: Schema drift report with list of affected fields and a patched translator that no longer depends on the shim.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures so the plan contract catches ordering regressions

Update fixtures to reflect the corrected schema from RN-101 so the plan-contract tests pass and catch future ordering regressions.

Bounded deliverable: Updated fixture files and passing plan_contract.py tests under the new schema.

Evidence:
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind a kill switch

Ship the user-visible dashboard summary behind a kill switch once the schema is stable and fixtures are backfilled.

Bounded deliverable: Dashboard summary feature flag with kill switch and canary coverage.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist after canary proof

Document the rollout and rollback shape now that the dashboard summary is proven in canary.

Bounded deliverable: Updated runbook and launch checklist reflecting the new summary path.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the corrected schema produced by the translator; without the schema audit the fixtures would encode the wrong step ids.
- `RN-101` before `RN-103`: Dashboard summary consumes the step-id schema from the translator; shipping before schema is stable risks rendering corrupted step order to users.
- `RN-102` before `RN-103`: Dependency map explicitly states the dashboard must not ship until fixtures reflect the new schema; otherwise plan_contract tests cannot catch ordering regressions.
- `RN-103` before `RN-104`: Runbook depends on knowing the rollout and rollback shape of the dashboard summary; it must wait until canary proves the feature.

## Primary Risk

Dashboard renders incorrect step ordering to users if the translator schema drift is not audited before the dashboard summary is enabled.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Complete RN-101 schema audit and remove the legacy shim before enabling the dashboard (RN-103).
- Backfill fixtures (RN-102) so plan_contract tests catch ordering regressions before canary.
- Ship the dashboard behind a kill switch so it can be disabled immediately if bad ordering appears in canary.

## Assumption Ledger

- Canary environment availability [missing]: No evidence in repo_inventory confirms a canary environment exists or is ready for RN-103 rollout testing.
- Legacy shim removal scope [missing]: Release notes do not specify whether the shim is removed in RN-101 or deferred to a later release; the plan assumes it is removed as part of the schema audit.

