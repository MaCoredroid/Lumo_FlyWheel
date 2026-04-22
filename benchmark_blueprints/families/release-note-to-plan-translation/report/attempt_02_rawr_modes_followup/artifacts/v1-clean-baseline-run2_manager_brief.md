# Release Plan Brief

- Variant: `v1-clean-baseline`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before shim removal

Use the legacy parser shim as a safety net while explicitly surfacing the translator's current step-id drift, so the team knows the real schema delta before any user-visible path depends on it.

Bounded deliverable: A confirmed step-id schema delta and a go or no-go decision for shim removal, without enabling the dashboard summary yet.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-graph fixtures for the new schema

Update the dependency-graph fixtures immediately after the schema audit so the plan contract fails on bad ordering and on newly introduced step ids instead of letting the shim mask regressions.

Bounded deliverable: Fixture coverage that reflects the audited step-id schema and makes ordering regressions visible in contract validation.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the dashboard summary behind a kill switch in canary

Expose the translated release-plan dashboard summary only after the schema and fixtures agree, and keep rollout constrained behind a kill switch because the dashboard is already user-visible and canary coverage is incomplete for bad dependency ordering.

Bounded deliverable: A canary-only dashboard summary rollout that can be disabled quickly if ordering or schema issues appear.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update operator runbook and launch checklist after canary proof

Finalize operator guidance only after the canary establishes the real rollout and rollback shape, so the checklist documents the proven path instead of freezing stale launch instructions.

Bounded deliverable: An operator runbook and launch checklist aligned to the validated canary and rollback procedure.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

## Dependency Notes

- `RN-101` before `RN-102`: The fixture backfill has to target the audited step-id schema; otherwise the contract will either encode the wrong ids or keep relying on the shim-hidden drift.
- `RN-102` before `RN-103`: The dashboard summary must not ship until dependency-graph fixtures reflect the new schema and can catch bad ordering regressions.
- `RN-103` before `RN-104`: The runbook only stabilizes after canary proves the real rollout and rollback path for the kill-switched dashboard summary.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and fixtures are backfilled, users can see a release dashboard that renders the translator's wrong order as if it were valid.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep the first milestone limited to exposing and deciding on the real translator schema delta before any launch work.
- Require dependency-graph fixture updates before the dashboard canary so ordering regressions are caught in the contract path.
- Use the kill switch during canary so any user-visible ordering error can be rolled back before operator documentation is finalized.

## Assumption Ledger

- Legacy shim consumers [to_verify]: The available evidence shows the shim masks local smoke drift, but it does not enumerate every caller that still depends on the legacy parser path.
- Bad-ordering canary coverage [observed]: Canary coverage exists for the dashboard summary, but the repo inventory explicitly says there is no bad dependency ordering coverage yet.
- Rollback acceptance criteria [missing]: The workspace does not define the exact rollback signal or operator threshold that should trigger disabling the dashboard kill switch during canary.

