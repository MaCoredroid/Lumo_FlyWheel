# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before touching exposure paths

Start by reconciling the frozen notes with the current translator contract and keep the legacy parser shim in place until the drift is understood, because the in-progress patch currently assumes the schema is already frozen when repo evidence says it is not.

Bounded deliverable: A documented schema audit with explicit freeze criteria and shim-removal exit conditions.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`

### 2. RN-102 — Backfill dependency-order fixtures and rollback-sensitive coverage

Once the schema target is known, add the dependency-graph fixtures and regression checks that catch hidden reordering so later rollout work is validated against the right contract instead of today’s unguarded behavior.

Bounded deliverable: Fixture-backed regression coverage that fails on dependency-order drift and exposes rollback-sensitive gaps.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the translated dashboard summary only as a kill-switched canary

After schema drift and ordering guards are under control, expose the translated release-plan summary behind a kill switch so the user-visible dashboard can be exercised without committing to a broad launch path.

Bounded deliverable: A kill-switched canary of the translated dashboard summary on the stabilized contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Refresh the operator runbook after canary behavior is proven

Update the operator checklist and launch guidance only after the canary path is proven, so the runbook reflects actual summary behavior rather than assumptions carried forward from the frozen notes.

Bounded deliverable: A runbook and launch checklist aligned to the proven canary workflow and kill-switch expectations.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Make an explicit keep-or-drop decision on the abandoned rollout draft

Treat the transferred engineer’s draft as optional cleanup after the safe path is clear, because it shares the unresolved schema and fixture dependencies and still lacks rollback coverage evidence.

Bounded deliverable: A documented decision on whether the abandoned draft is discarded or respecified for follow-on work.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

## Dependency Notes

- `RN-101` before `RN-102`: The fixture backfill has to target the audited translator schema; otherwise it will lock in the same false frozen-schema assumption called out in the in-progress patch.
- `RN-102` before `RN-103`: The dashboard summary is already user-visible, so ordering-regression coverage needs to exist before any canary exposure of translated plans.
- `RN-103` before `RN-104`: Operator guidance should be written from the kill-switched canary path that actually worked, not from pre-canary expectations.
- `RN-103` before `RN-105`: Repo evidence marks the draft as optional cleanup on the same unresolved path, so it should only be reconsidered after the safer dashboard path is proven.

## Primary Risk

If the user-visible dashboard summary or the abandoned rollout draft advances before schema drift and dependency-order coverage are closed, operators can be shown a translated implementation order that is wrong and cannot be rolled back safely.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Keep the legacy parser shim in place until the schema audit produces explicit freeze criteria.
- Add dependency-order fixtures before any canary so hidden reordering regressions fail in test rather than on the live dashboard.
- Defer any decision on reviving the abandoned draft until the kill-switched safe path is proven.

## Assumption Ledger

- Translator schema drift scope [observed]: The in-progress patch explicitly assumes the schema is frozen, and repo evidence says that assumption is false today.
- Canary target and success criteria [to_verify]: The notes require proving the summary path in canary, but the repo evidence does not identify the target environment or pass criteria.
- Rollback behavior of the abandoned draft [missing]: The test inventory says the draft never added rollback coverage, so the cost and safety of finishing it cannot be assessed from the current evidence.

