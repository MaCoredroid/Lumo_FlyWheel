# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit schema drift before touching rollout paths

Start by auditing translator schema drift and explicitly keeping the legacy parser shim in place until the schema is actually stable. This is the smallest milestone that removes the false assumption already embedded in the abandoned patch.

Bounded deliverable: A schema-drift audit with a keep-or-remove decision for the legacy parser shim, without enabling any user-visible summary path yet.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`

### 2. RN-102 — Backfill dependency-order fixtures and rollback coverage

Once the audited contract is known, add dependency-graph fixtures and regression checks that catch hidden dependency reordering. This closes the explicit test gap before any canary or dashboard exposure.

Bounded deliverable: Coverage that fails when dependency ordering changes silently and a plan contract that reflects the audited schema.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the dashboard summary behind the kill switch

Only after schema and fixture work are stable should the translated release-plan dashboard summary be exposed behind the kill switch. The dashboard is already user-visible, so this step must be a controlled canary rather than a broad launch.

Bounded deliverable: A canary summary path guarded by the kill switch, using the audited schema and the new dependency-order regression coverage.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update the operator runbook after canary proof

Refresh the operator runbook and launch checklist only after the canary path is proven. This prevents documenting a rollout sequence that still reflects stale assumptions or pre-canary behavior.

Bounded deliverable: An operator runbook and launch checklist that match the proven canary sequence instead of the earlier draft order.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Review or retire the abandoned rollout draft last

Treat the abandoned rollout draft as optional cleanup after the safe path is clear. The draft is stale, its owner transferred teams, and it depends on the same unresolved schema and fixture work as the dashboard path.

Bounded deliverable: A keep-or-drop decision on the abandoned draft, with no attempt to finish it before the safe rollout path is settled.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/in_progress_patch/README.md`

## Dependency Notes

- `RN-101` before `RN-102`: The fixture contract should lock the audited schema, not codify the current false assumption that the schema is already frozen.
- `RN-102` before `RN-103`: The user-visible dashboard summary should not enter canary until dependency reordering is caught by tests.
- `RN-103` before `RN-104`: Operators need runbook steps that describe the proven canary path instead of a pre-canary guess.
- `RN-103` before `RN-105`: The abandoned draft is optional cleanup and should only be judged after the safe dashboard path is clear.

## Primary Risk

If the team enables the dashboard summary or resumes the abandoned rollout draft before auditing schema drift and adding ordering-regression fixtures, the live release dashboard can present the wrong execution order to users with no rollback-aware coverage to catch it.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Keep the legacy parser shim and finish the schema-drift audit before any user-visible rollout work.
- Backfill dependency-graph fixtures and rollback-aware regression coverage before canarying the summary path.
- Leave the transferred engineer's rollout draft as end-of-path cleanup unless the canary path proves it is still useful.

## Assumption Ledger

- Schema is not frozen today [observed]: The in-progress patch explicitly assumes the translator schema is already frozen, and repo evidence says that assumption is false.
- Canary success criteria [to_verify]: The release notes require the summary path to be proven in canary, but the exact exit metrics are not provided in the workspace evidence.
- Shifted release or incident context [missing]: No release_context or incident_context directory is present, so there is no newer objective or rollback note beyond the provided release notes and repo inventory.

