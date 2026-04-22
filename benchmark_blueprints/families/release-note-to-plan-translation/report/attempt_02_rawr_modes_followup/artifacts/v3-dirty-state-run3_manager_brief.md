# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift before touching rollout paths

Start by reconciling the current translator schema with the still-required legacy parser shim and explicitly record that shim removal is blocked until the schema is actually stable. This is the smallest meaningful milestone because the in-progress draft already assumes a frozen schema that the repo evidence says does not exist.

Bounded deliverable: A confirmed schema-drift decision point that keeps the shim in place and invalidates stale rollout assumptions.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`

### 2. RN-102 — Backfill dependency-order fixtures and regression coverage

Once the schema boundary is clear, add the dependency-graph fixtures and ordering checks that catch hidden reordering before any user-visible summary work resumes. The repo evidence says the current regression suite still misses this exact failure mode, so this must land before rollout or canary steps.

Bounded deliverable: A fixture-backed regression layer that fails when translated plan ordering drifts or hidden dependencies reorder.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the translated dashboard summary behind a kill switch

Only after schema and ordering guardrails are stable should the translated release-plan dashboard summary be reintroduced in a kill-switched canary. The dashboard is already user-visible, so the safe path is a limited exposure check rather than a broad launch.

Bounded deliverable: A kill-switched canary proving the translated summary path on top of the stabilized schema and fixture contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update operator runbook and launch checklist after canary proof

Revise the operator runbook and launch checklist only after the canary demonstrates that the summary path works under the kill switch. This keeps operational guidance aligned with the path that was actually proven instead of documenting an unverified flow.

Bounded deliverable: Operator guidance that matches the canary-proven summary path and launch sequencing.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Review the abandoned rollout draft as optional cleanup

Treat the half-finished rollout draft as stale optional cleanup after the safe path is already clear. Both the dependency map and the patch README say it sits on unresolved schema and fixture assumptions, so finishing it earlier would just reintroduce the same invalid premise.

Bounded deliverable: A keep-or-discard decision on the abandoned draft after the stable rollout path is already established.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/in_progress_patch/README.md`

## Dependency Notes

- `RN-101` before `RN-102`: Ordering fixtures need the current schema boundary and shim decision to lock the correct contract.
- `RN-102` before `RN-103`: The user-visible dashboard summary should not canary until hidden dependency reordering is covered by regression fixtures.
- `RN-103` before `RN-104`: The runbook and checklist must document the summary path that actually passed canary under the kill switch.
- `RN-101` before `RN-105`: The abandoned draft assumes a frozen schema, so it cannot be reviewed honestly until that assumption is resolved.
- `RN-102` before `RN-105`: The draft shares the same unresolved fixture and ordering-coverage gap as the dashboard rollout work.

## Primary Risk

If dashboard or draft-rollout work starts before schema drift is audited and ordering fixtures exist, the still-visible release dashboard can expose a misordered implementation summary with no rollback-behavior coverage, causing operators to follow the wrong launch sequence.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Keep the legacy parser shim in place until the schema audit closes the current drift.
- Backfill dependency-order fixtures before any dashboard canary or rollout-draft revival.
- Limit translated summary exposure to a kill-switched canary before updating operator guidance.

## Assumption Ledger

- Dashboard exposure [observed]: The repo state says the release dashboard is still user-visible, so user-facing sequencing mistakes matter immediately.
- Draft status [observed]: The in-progress patch is a transferred-team draft that incorrectly assumes the translator schema is already frozen.
- Kill-switch implementation details [to_verify]: The release notes require a kill-switched canary, but the repo evidence here does not show where that switch currently lives.
- Rollback acceptance criteria [missing]: The test inventory says rollback behavior never received coverage, and no incident or release context file here defines the acceptance criteria.

