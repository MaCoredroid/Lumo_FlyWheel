# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Audit and document schema drift in the translator before removing the legacy parser shim. This reduces ambiguity about what the translator produces.

Bounded deliverable: Schema drift audit report identifying differences between current translator output and expected step-id schema

Evidence:
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Update dependency-graph fixtures so the plan contract catches ordering regressions when new step ids are introduced.

Bounded deliverable: Updated fixtures in tests/plan_contract.py that pass with new step ids

Evidence:
- `repo_inventory/test_inventory.md`
- `release_notes/release_notes_2026_04.md`

### 3. RN-103 — Enable dashboard summary behind kill switch

Enable the translated release-plan dashboard summary behind a kill switch after the schema is stable and fixtures are in place.

Bounded deliverable: Dashboard summary feature flag enabled in canary with kill switch available

Evidence:
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Update the operator runbook and launch checklist after the summary path is proven in canary.

Bounded deliverable: Updated runbook documenting rollout and rollback procedures

Evidence:
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Translator schema must be audited and stabilized before fixtures can be correctly updated
- `RN-102` before `RN-103`: Dashboard summary depends on step-id schema and fixtures; fixtures must reflect new schema before dashboard can render correctly
- `RN-103` before `RN-104`: Runbook requires knowledge of rollout and rollback shape which is only known after dashboard is proven in canary

## Primary Risk

Dashboard renders incorrect or unstable step ordering to users if schema drift is not audited and fixtures are not updated before enabling the summary

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Audit schema drift first (RN-101) to understand what the translator actually produces
- Backfill fixtures (RN-102) before enabling dashboard to ensure plan contract catches ordering issues
- Use kill switch (RN-103) to disable dashboard quickly if issues appear in canary
- Keep legacy parser shim in place until schema audit is complete

## Assumption Ledger

- Legacy parser shim removal [missing]: No explicit release note for removing the shim; audit (RN-101) must inform whether removal is safe
- Kill switch availability [to_verify]: Kill switch infrastructure needs verification for RN-103
- Canary environment [to_verify]: Canary deployment path needs verification for RN-103 and RN-104 validation

