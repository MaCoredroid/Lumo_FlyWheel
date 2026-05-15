# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Identify schema drift before removing shim

Bounded deliverable: Schema drift audit report

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Add fixtures for new schema

Bounded deliverable: Updated fixtures

Evidence:
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable dashboard summary

Enable dashboard behind kill switch

Bounded deliverable: Dashboard in canary

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update runbook

Document rollout procedures

Bounded deliverable: Updated runbook

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Audit before backfill
- `RN-102` before `RN-103`: Fixtures before dashboard
- `RN-103` before `RN-104`: Canary before runbook

## Primary Risk

Dashboard shows wrong order if schema drift not resolved

Evidence:
- `repo_inventory/repo_state.md`

Mitigations:
- Audit first
- Fixtures second
- Kill switch third

## Assumption Ledger

- drift_scope [missing]: Unknown until audit

