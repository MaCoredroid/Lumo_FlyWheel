# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101-schema-audit`

## Ordered Plan

### 1. RN-101-schema-audit — Audit translator schema drift and freeze the step identifiers

Start by reconciling the rewritten translator schema with the legacy parser shim so the release plan uses one stable contract before any user-facing rollout work proceeds.

Bounded deliverable: A signed-off schema audit that names any remaining shim dependency and confirms the canonical step identifiers for downstream consumers.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 2. RN-102-fixture-backfill — Backfill dependency-graph fixtures to lock the ordering contract

Once the schema is stable, add the missing dependency-graph fixtures so the plan contract can catch ordering regressions and step-id drift before rollout.

Bounded deliverable: Fixture coverage that exercises the dependency chain used by the translated release-plan summary and protects against step-id drift.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103-dashboard-canary — Enable the dashboard summary behind a kill switch for canary users

Only after the schema audit and fixture backfill are complete should the translated dashboard summary be exposed behind a kill switch and proven on a canary path.

Bounded deliverable: A canary-ready dashboard summary path that can be disabled quickly if the translated ordering still diverges from the verified contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104-runbook-update — Update the operator runbook and launch checklist after canary proof

Refresh operational guidance only after the canary demonstrates that the kill-switched summary path is the correct launch path for the current release objective.

Bounded deliverable: An updated runbook and launch checklist aligned with the proven canary sequence rather than the pre-rewrite notes.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105-cache-followup — Revisit the January cache experiment only as a post-path optimization

Keep the January cache experiment last because the notes and repo inventory both say it predates the schema rewrite and does not fix the current dashboard ordering problem.

Bounded deliverable: A documented decision on whether any cache idea still helps after the release summary path is stable and launched safely.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101-schema-audit` before `RN-102-fixture-backfill`: Fixture backfill must target the audited schema and stable step identifiers or it will encode the same drift now breaking the plan contract.
- `RN-102-fixture-backfill` before `RN-103-dashboard-canary`: The dashboard rollout should wait for dependency fixtures so ordering regressions are caught before users see the translated summary.
- `RN-103-dashboard-canary` before `RN-104-runbook-update`: Operational guidance should describe the path that actually survived canary, not an unproven rollout sequence.
- `RN-104-runbook-update` before `RN-105-cache-followup`: Performance tuning is secondary until the current dashboard objective and operator workflow are stable.

## Primary Risk

If the dashboard summary is rolled out before the schema audit and fixture backfill, users can see the wrong implementation order in the release dashboard while operators follow a checklist that reflects an invalid sequence.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `release_notes/stale_note_jan_2026.md`

Mitigations:
- Freeze the translator schema and remaining shim dependency before any dashboard exposure.
- Backfill dependency fixtures so step-id drift and ordering regressions fail before rollout.
- Use the kill switch to constrain the dashboard summary to canary users until the sequence is proven.

## Assumption Ledger

- Current objective [observed]: The repo state says the user-visible dashboard summary is still the active release target.
- Shim removal scope [to_verify]: The notes require auditing schema drift before removing the legacy parser shim, but they do not confirm whether shim removal itself is in this release cut.
- Canary exit criteria [missing]: The workspace does not include explicit canary success metrics or rollback thresholds for promoting the kill-switched dashboard summary.

