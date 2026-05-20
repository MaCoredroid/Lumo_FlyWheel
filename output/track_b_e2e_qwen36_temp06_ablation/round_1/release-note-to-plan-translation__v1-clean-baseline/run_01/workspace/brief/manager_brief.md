# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Audit the translator schema to surface drift that the legacy parser shim currently masks. This is a bounded investigation that de-risks all downstream work without touching user-visible paths.

Bounded deliverable: Schema drift report with a concrete list of fields that differ between legacy and translated schemas.

Evidence:
- `repo_inventory/repo_state.md`
- `release_notes/release_notes_2026_04.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the new schema

Backfill the dependency-graph fixtures so the plan contract test catches ordering regressions under the new step-id schema. This must happen after the schema audit so the fixtures reflect the corrected field names.

Bounded deliverable: Updated fixtures that make tests/plan_contract.py pass with new step ids.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind a kill switch

Enable the dashboard summary behind a kill switch now that the schema is stable and fixtures are in place. The dashboard is user-visible, so it must only render after the translator output is correct.

Bounded deliverable: Dashboard summary feature-flagged behind a kill switch, validated in canary.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist

Finalize the operator runbook and launch checklist after the summary path is proven in canary. The runbook depends on knowing the rollout and rollback shape, which is only known after RN-103.

Bounded deliverable: Updated runbook and launch checklist reflecting proven rollout/rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`
- `release_notes/release_notes_2026_04.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the corrected schema; backfilling before the audit would encode the wrong field names.
- `RN-102` before `RN-103`: Dashboard summary must not ship until dependency-graph fixtures reflect the new schema, otherwise plan_contract.py fails.
- `RN-103` before `RN-104`: Runbook depends on knowing the rollout and rollback shape, which is only determined after the dashboard is proven in canary.

## Primary Risk

Dashboard renders steps in the wrong order, causing operators to execute release steps out of sequence.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep the dashboard behind a kill switch until canary validation passes (RN-103).
- Ensure plan_contract.py passes with the new fixtures before enabling the dashboard (RN-102).
- Audit schema drift first so the translator emits correct step ids (RN-101).

## Assumption Ledger

- Canary environment availability [missing]: No evidence in repo_inventory confirms a canary environment is ready for dashboard validation.
- Legacy parser shim removal timeline [to_verify]: RN-101 audits drift before shim removal, but the shim removal itself is not in scope for this plan.

