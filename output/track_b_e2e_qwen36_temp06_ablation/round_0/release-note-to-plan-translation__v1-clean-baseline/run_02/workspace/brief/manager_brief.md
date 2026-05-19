# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Identify and document all schema drift currently masked by the legacy parser shim. This de-risks the entire path by revealing the true state of the translator output before any downstream work begins.

Bounded deliverable: Schema drift audit report with concrete list of drifted fields; legacy shim remains in place.

Evidence:
- `repo_inventory/repo_state.md`
- `release_notes/release_notes_2026_04.md`

### 2. RN-102 — Backfill dependency-graph fixtures so plan contract catches ordering regressions

Update dependency-graph fixtures to reflect the corrected schema from RN-101 so the plan contract test suite passes and catches ordering regressions.

Bounded deliverable: Updated fixtures that make tests/plan_contract.py pass with new step IDs.

Evidence:
- `repo_inventory/test_inventory.md`
- `release_notes/release_notes_2026_04.md`

### 3. RN-103 — Enable translated release-plan dashboard summary behind a kill switch

Ship the dashboard summary behind a kill switch now that the schema is stable and fixtures are backfilled. The dashboard consumes the step-id schema from the translator, so it must wait for RN-101 and RN-102.

Bounded deliverable: Dashboard summary enabled behind kill switch in canary only.

Evidence:
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `release_notes/release_notes_2026_04.md`

### 4. RN-104 — Update operator runbook and launch checklist after summary path is proven in canary

Finalize the runbook and launch checklist once the dashboard summary rollout and rollback shape are known from canary validation.

Bounded deliverable: Updated runbook and launch checklist reflecting proven rollout/rollback procedures.

Evidence:
- `repo_inventory/dependency_map.md`
- `release_notes/release_notes_2026_04.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixtures must reflect the corrected schema; backfilling against un-audited drift would bake errors into the test suite.
- `RN-101` before `RN-103`: The dashboard summary consumes the step-id schema produced by the translator; shipping it before the schema is stable risks rendering garbage to users.
- `RN-102` before `RN-103`: The dashboard must not ship until dependency-graph fixtures reflect the new schema, otherwise plan_contract.py cannot catch ordering regressions.
- `RN-103` before `RN-104`: The runbook only stabilizes after the rollout and rollback shape are known from canary validation of the dashboard summary.

## Primary Risk

The release dashboard renders incorrect or garbled step ordering to users because the translator schema drift was not audited before the dashboard summary was enabled.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep RN-101 as the first milestone to audit drift before any downstream work.
- Ship RN-103 behind a kill switch so the dashboard can be disabled instantly if bad data appears.
- Ensure RN-102 fixtures are updated before RN-103 so plan_contract.py catches ordering regressions in CI.

## Assumption Ledger

- Kill switch implementation [missing]: No evidence in repo_inventory that a kill switch for the dashboard summary already exists; RN-103 assumes one can be added or is present.
- Canary environment readiness [missing]: test_inventory.md mentions canary-only coverage for the dashboard summary, but there is no evidence the canary environment is configured to validate the new schema path.

