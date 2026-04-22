# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing the legacy parser shim

Establish the post-shim step-id schema and expose any drift that local smoke runs currently hide so downstream work is anchored on the real contract.

Bounded deliverable: A confirmed translator schema baseline and a concrete list of drift findings that must be honored by tests and downstream consumers.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the new step-id schema

Update dependency-graph fixtures immediately after the schema audit so the plan contract can catch ordering regressions before any user-visible summary path is enabled.

Bounded deliverable: Fixture coverage that reflects the audited schema and fails when translated plans introduce bad dependency ordering.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable the translated release-plan dashboard summary behind a kill switch in canary

After the schema and fixtures are aligned, expose the dashboard summary only behind the kill switch and prove the summary path in canary before any broader rollout.

Bounded deliverable: A canary-validated dashboard summary path that consumes the audited schema while retaining an immediate rollback switch.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update the operator runbook and launch checklist after canary proves rollout shape

Document the operational procedure only after canary clarifies the real rollout and rollback shape so the checklist matches the shipped behavior.

Bounded deliverable: An operator runbook and launch checklist that reflect the proven canary flow, kill-switch usage, and rollback expectations.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: The fixture backfill has to target the audited post-shim step-id contract; otherwise the tests will be updated against the wrong schema.
- `RN-102` before `RN-103`: The dashboard summary consumes translator step ids and must not ship until dependency-graph fixtures catch ordering regressions under the new schema.
- `RN-103` before `RN-104`: The runbook and launch checklist depend on the actual canary rollout and rollback behavior, including kill-switch operation.

## Primary Risk

If the dashboard summary is enabled before schema drift is exposed and fixtures are backfilled, users can see a release plan rendered in the wrong order without guardrails catching the dependency mistake first.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Make the schema-drift audit the first milestone so downstream steps use the real translator contract.
- Backfill dependency fixtures before the dashboard canary so contract tests fail on ordering regressions.
- Keep the dashboard behind the kill switch during canary so operators can roll back quickly if ordering defects surface.

## Assumption Ledger

- Schema audit outcome [observed]: Current evidence confirms the legacy parser shim masks schema drift, so the first milestone must expose the real translator contract.
- Dashboard dependency [observed]: Current repo evidence confirms the dashboard summary consumes translator step ids and should follow fixture alignment.
- Canary exit criteria [missing]: The workspace does not document exact canary success thresholds or rollback triggers for the dashboard summary, so the runbook details cannot be finalized yet.

