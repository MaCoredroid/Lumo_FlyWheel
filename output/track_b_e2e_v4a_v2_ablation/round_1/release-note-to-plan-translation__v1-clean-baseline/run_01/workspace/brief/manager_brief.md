# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Identify all schema drift between translator output and expected step-id schema while the legacy parser shim is still in place, since the shim masks drift in local smoke runs.

Bounded deliverable: Documented list of schema drift issues and decision on shim removal readiness.

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures for plan contract

Add dependency-graph fixtures so the plan contract test catches ordering regressions when new step ids are introduced.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that pass with new schema.

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind kill switch

Enable the dashboard summary feature behind a kill switch now that the schema is stable and fixtures validate ordering.

Bounded deliverable: Dashboard summary visible in canary behind feature flag.

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update operational documentation with rollout and rollback procedures once the summary path is proven in canary.

Bounded deliverable: Updated runbook and launch checklist with proven procedures.

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema drift must be audited before fixtures can be correctly backfilled; the legacy shim masks drift.
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema.
- `RN-103` before `RN-104`: Runbook only stabilizes after rollout and rollback shape are known from canary validation.

## Primary Risk

Dashboard renders incorrect step ordering to users if schema drift is not resolved before enabling the summary.

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Complete RN-101 audit before any schema changes
- Pass RN-102 fixture tests to validate ordering contract
- Keep RN-103 behind kill switch for canary-only exposure

## Assumption Ledger

- incident_context [missing]: No incident_context directory present; assuming no prior rollback invalidated the release note order.
- release_context [missing]: No release_context directory present; assuming original release notes remain valid and unstale.

