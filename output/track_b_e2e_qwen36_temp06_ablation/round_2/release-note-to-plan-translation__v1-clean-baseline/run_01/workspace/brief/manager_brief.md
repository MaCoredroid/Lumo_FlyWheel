# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and remove legacy parser shim

Audit the translator for schema drift that the legacy parser shim currently masks, then remove the shim so drift is visible in local smoke runs.

Bounded deliverable: Schema drift audit report and shim removal patch

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Backfill dependency-graph fixtures so the plan contract catches ordering regressions once the new schema is in place.

Bounded deliverable: Dependency-graph fixture set covering new schema

Evidence:
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable translated release-plan dashboard summary

Enable the dashboard summary behind a kill switch now that the schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature behind kill switch

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Finalize the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook and launch checklist

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the corrected schema; schema audit must complete first.
- `RN-101` before `RN-103`: Dashboard consumes the translator step-id schema; schema must be stable before enabling.
- `RN-102` before `RN-103`: Dashboard must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after the summary rollout and rollback shape are known from canary.

## Primary Risk

Dashboard renders incorrect step order to users if schema drift is not resolved before the summary is enabled.

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Keep kill switch on dashboard summary until canary proves stability
- Gate dashboard behind fixture validation so missing fixtures block deployment
- Run schema-drift audit in CI so regressions are caught before merge

## Assumption Ledger

- Canary environment readiness [missing]: No evidence in repo_inventory confirms a canary environment exists or is configured for the dashboard summary.
- Kill switch implementation [missing]: Assume a kill switch mechanism already exists for the dashboard summary feature; no config evidence found.
- Legacy shim removal scope [missing]: Assume the shim is a single removable module; no code-level evidence confirms coupling to other components.

