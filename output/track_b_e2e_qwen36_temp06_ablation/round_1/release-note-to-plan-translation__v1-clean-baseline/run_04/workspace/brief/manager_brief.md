# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and remove legacy parser shim

Audit the translator for schema drift while the legacy parser shim is still in place, then remove the shim. The shim currently masks drift in local smoke runs, so this must happen first to surface the real schema state.

Bounded deliverable: Schema drift audit report with shim removed from the codebase.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the new schema

Backfill dependency-graph fixtures so the plan contract test catches ordering regressions under the new step-id schema. This must precede the dashboard because the dashboard must not ship until fixtures reflect the new schema.

Bounded deliverable: Updated fixtures that make tests/plan_contract.py pass with new step ids.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind a kill switch

Enable the dashboard summary behind a kill switch now that the schema is stable and fixtures are backfilled. The dashboard is user-visible and renders whatever order the translator emits, so the kill switch provides a safety net.

Bounded deliverable: Dashboard summary enabled behind kill switch with canary coverage.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the runbook and launch checklist after the summary path is proven in canary. The runbook only stabilizes once the rollout and rollback shape are known from canary validation.

Bounded deliverable: Updated runbook and launch checklist reflecting proven canary rollout/rollback procedures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the new schema discovered by the audit; schema drift must be audited first so the correct step ids are known before backfilling.
- `RN-101` before `RN-103`: Dashboard consumes the translator step-id schema; schema must be stable before enabling the dashboard.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema; without fixtures, plan_contract.py fails on new step ids.
- `RN-103` before `RN-104`: Runbook only stabilizes after the summary path is proven in canary and the rollout/rollback shape is known.

## Primary Risk

If the dashboard summary (RN-103) is enabled before the schema audit (RN-101) and fixture backfill (RN-102) are complete, the user-visible release dashboard will render incorrect step ordering based on a drifted schema with no test safety net.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Enable dashboard behind a kill switch (RN-103) so it can be disabled immediately if bad ordering appears.
- Complete schema audit and fixture backfill before toggling the dashboard on.
- Run plan_contract.py tests to verify ordering before canary deployment.

## Assumption Ledger

- Canary deployment configuration [missing]: No evidence of the canary rollout config or traffic-split thresholds; RN-104 assumes canary proof is available but no canary infrastructure details are present in the repo inventory.
- Kill switch implementation [missing]: RN-103 mentions a kill switch but no implementation or feature-flag infrastructure is documented in the repo inventory. Assuming a feature flag system exists to gate the dashboard.

