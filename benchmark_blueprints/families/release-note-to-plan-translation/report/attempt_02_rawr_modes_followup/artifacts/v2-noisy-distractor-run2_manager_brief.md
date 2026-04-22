# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before shim removal

Start by reconciling the translator schema rewrite with the legacy parser shim so the team knows the stable contract before any downstream rollout work.

Bounded deliverable: A schema-audit decision that confirms the current translator contract and whether the legacy parser shim can stay or be removed.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 2. RN-102 — Backfill dependency-graph fixtures and ordering checks

Once the schema contract is audited, backfill fixtures and plan-contract coverage so ordering regressions and step-id drift are caught before the dashboard path is exercised.

Bounded deliverable: Fixture coverage that reflects the audited schema and guards against ordering regressions, including the stale-note path ranking last.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the translated dashboard summary behind a kill switch

Expose the translated release-plan dashboard summary only after the schema and fixtures are stable, and keep it gated so canary behavior can be proven safely.

Bounded deliverable: A kill-switched canary of the dashboard summary that uses the audited schema and validated ordering fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update the operator runbook after canary proof

Operator-facing runbook and launch checklist changes should follow the canary proof so the docs describe the real gated workflow instead of a still-moving draft path.

Bounded deliverable: An updated operator runbook and launch checklist that reflect the proven canary path and summary guardrails.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Review the January cache experiment only as a later optimization

Treat the January cache experiment as stale context and revisit it only after the release dashboard objective is stable, because it does not resolve the current ordering problem.

Bounded deliverable: A deferred follow-up decision on whether any cache ideas remain useful after the dashboard path is working.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture backfills need the audited translator schema so tests do not lock in the wrong step identifiers or shim behavior.
- `RN-102` before `RN-103`: The dashboard summary is user-visible, so ordering fixtures and regression checks must be in place before any canary exposure.
- `RN-103` before `RN-104`: The runbook and launch checklist should document the proven canary workflow, not a pre-canary draft that may still change.
- `RN-104` before `RN-105`: The cache experiment is explicitly stale and should stay behind the current release objective and its operator rollout work.

## Primary Risk

If the dashboard summary is enabled before schema drift is audited and ordering fixtures are backfilled, users will see a release-plan summary that can misorder steps or surface stale January work as current scope.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `release_notes/stale_note_jan_2026.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Hold the dashboard summary behind the kill switch until the schema audit resolves the legacy parser shim boundary.
- Backfill dependency-graph fixtures and close the step-id drift gap before canary exposure.
- Keep the January cache experiment ranked last and explicitly mark it as stale in planning and rollout review.

## Assumption Ledger

- Current release objective [observed]: The repo state says the user-visible dashboard summary is the active release objective.
- Ordering guard coverage [to_verify]: The test inventory shows step-id drift failures and no assertion yet that stale experiments are ranked last.
- Canary exit criteria [missing]: The workspace does not define the exact success signal that lets the team move from kill-switched canary to operator checklist updates.

