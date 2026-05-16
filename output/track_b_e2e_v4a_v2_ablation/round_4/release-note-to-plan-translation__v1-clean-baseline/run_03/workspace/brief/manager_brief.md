# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Identify schema drift in the translator before removing the legacy parser shim. This reveals the actual step-id schema that downstream components depend on.

Bounded deliverable: Documented schema drift report with concrete step-id changes

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update fixtures to reflect the new schema so the plan contract test catches ordering regressions.

Bounded deliverable: Updated fixtures that pass tests/plan_contract.py with new step ids

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after schema stability is confirmed.

Bounded deliverable: Dashboard summary enabled in canary with kill switch for rollback

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Document rollout and rollback procedures once the summary path is proven in canary.

Bounded deliverable: Updated runbook with validated rollout/rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Schema audit must precede fixture updates; fixtures need to reflect the actual drift discovered in the audit
- `RN-102` before `RN-103`: Dashboard summary consumes step-id schema and must not ship until fixtures reflect the new schema
- `RN-103` before `RN-104`: Runbook requires known rollout/rollback shape which only becomes available after dashboard is proven in canary

## Primary Risk

Dashboard renders incorrect step ordering to users if schema drift is not audited before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Complete RN-101 schema audit before any dashboard work
- Update fixtures in RN-102 to catch ordering regressions in tests
- Enable dashboard behind kill switch (RN-103) for safe rollback

## Assumption Ledger

- schema_drift_scope [missing]: Exact magnitude of schema drift is unknown until RN-101 audit completes; plan assumes drift exists based on legacy shim masking behavior

