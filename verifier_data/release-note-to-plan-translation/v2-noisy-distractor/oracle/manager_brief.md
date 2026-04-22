# Release Plan Brief

- Variant: `v2-noisy-distractor`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift

Freeze the step-id schema and map the hidden dependency edges before any downstream rollout work.

Bounded deliverable: Freeze the exact prerequisite and produce one reviewable artifact for this step only.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 2. RN-102 — Backfill dependency-graph fixtures

Extend plan-contract fixtures so the new dependency edges fail loudly before release.

Bounded deliverable: Land the minimum scoped change plus the directly blocked follow-on evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 3. RN-103 — Enable gated dashboard summary rollout

Turn on the translated release-plan summary behind a kill switch once prerequisites are stable.

Bounded deliverable: Land the minimum scoped change plus the directly blocked follow-on evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update operator runbook and launch checklist

Document the new summary workflow and the rollback path once the rollout path is known.

Bounded deliverable: Land the minimum scoped change plus the directly blocked follow-on evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Revive deprecated cache experiment

An older perf note suggests a cache path, but the note predates the current translator flow and should not lead the plan.

Bounded deliverable: Land the minimum scoped change plus the directly blocked follow-on evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

## Dependency Notes

- `RN-101` before `RN-103`: Repo-state evidence says this prerequisite must land first to avoid shipping the wrong user-visible order.
- `RN-102` before `RN-103`: Repo-state evidence says this prerequisite must land first to avoid shipping the wrong user-visible order.
- `RN-103` before `RN-104`: Repo-state evidence says this prerequisite must land first to avoid shipping the wrong user-visible order.

## Primary Risk

If the team ships the dashboard summary before the prerequisite ordering work is locked, users will see a wrong or misleading execution order during release rollout.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

Mitigations:
- Keep the summary behind a kill switch until prerequisites are green.
- Use the plan-contract fixtures as the rollout gate for dependency ordering.
- Update the operator runbook before broadening exposure.

## Assumption Ledger

- Exact dashboard traffic split for the kill switch ramp [to_verify]: Repo inventory names the gate but not the final percentage ladder.
- Whether any hidden customer workflow still depends on the legacy parser shim [missing]: Current notes imply hidden coupling but do not enumerate every downstream consumer.

