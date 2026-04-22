# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before removing legacy parser shim

Use the existing shim-masked smoke gap to identify the exact translator step-id/schema drift and lock the new contract before any downstream consumer work proceeds.

Bounded deliverable: A confirmed translator step-id/schema contract and a documented decision on whether the legacy parser shim can be removed without hiding drift.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the new plan contract

Update fixture coverage immediately after the schema audit so contract tests catch ordering regressions before any user-visible summary consumes the new translator output.

Bounded deliverable: Dependency-graph fixtures aligned to the audited step-id schema, with plan-contract coverage no longer failing on introduced step ids.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Enable the translated release-plan dashboard summary behind a kill switch

Only after the schema and fixtures are stable, expose the user-visible dashboard summary in canary with a kill switch so bad ordering can be rolled back quickly.

Bounded deliverable: A canary-gated dashboard summary path wired behind a kill switch, using the audited translator schema and updated fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update the operator runbook and launch checklist after canary proves the path

Document rollout and rollback instructions only once the canary behavior and rollback shape are known from the gated dashboard path.

Bounded deliverable: An operator runbook and launch checklist that reflect the actual canary rollout and rollback behavior of the translated summary path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfill has to target the audited translator step-id/schema contract; otherwise tests are updated against a moving or incorrect shape.
- `RN-102` before `RN-103`: The dashboard summary must not ship until dependency-graph fixtures reflect the new schema and ordering regressions are caught before users see them.
- `RN-103` before `RN-104`: The runbook depends on the observed canary rollout and rollback path, so documenting operations earlier would encode unproven steps.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and fixtures are backfilled, users will see a release plan ordered by incorrect translator output, with no reliable automated signal for bad dependency ordering.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep RN-101 as the first milestone so the translator contract is explicit before downstream work.
- Complete RN-102 before RN-103 so fixture-backed contract tests catch ordering regressions ahead of canary exposure.
- Ship RN-103 behind a kill switch so canary rollback remains immediate if user-visible ordering is wrong.

## Assumption Ledger

- Release objective stability [observed]: No release_context/ directory is present, so there is no newer objective override to supersede the frozen April release notes in this workspace.
- Prior rollback impact [observed]: No incident_context/ directory is present, so there is no local evidence of a rollback that would reorder RN-101 through RN-104.
- Canary exit criteria [missing]: The repo inventory states canary-only coverage exists, but it does not define the concrete success or rollback thresholds required before the runbook is finalized.

