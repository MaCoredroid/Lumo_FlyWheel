# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-graph fixtures first

Make the missing dependency-fixture coverage the first bounded milestone because the current objective shifted toward safety and rollback confidence, and the repo still lacks a customer-facing safety check until this backfill lands.

Bounded deliverable: Dependency-graph fixtures and contract coverage that catch ordering regressions before any wider rollout change.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 2. RN-101 — Audit translator schema drift after fixtures exist

Run the schema-drift audit once the ordering fixtures are in place so the audit is judged against the current safe-plan contract, while explicitly keeping legacy parser shim removal out of the first milestone.

Bounded deliverable: A schema-drift audit result and shim decision note tied to the updated ordering contract, not a full parser cutover.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `release_context/current_objective.md`

### 3. RN-103 — Gate dashboard summary behind kill switch

Only after fixture coverage and schema audit are both complete should the translated dashboard summary move behind the kill switch, because dashboard speed is no longer the dominant objective and the dashboard is already user-visible.

Bounded deliverable: A canary-ready dashboard summary path that can be disabled quickly if the translated ordering is wrong.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 4. RN-104 — Update operator runbook and launch checklist

Refresh the runbook and launch checklist after the canary proves the gated summary path so operators have rollback and guarded-change-window guidance aligned with the new safety-first objective.

Bounded deliverable: Operator instructions and launch checklist that reflect proven canary behavior and rollback posture.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`

### 5. RN-105 — Send enterprise change-window update last

Hold the enterprise communication until ordering guarantees, operator guidance, and rollback handling are all locked, because external change-window messaging is blocked on correct ordering guarantees rather than on dashboard availability.

Bounded deliverable: A customer-facing change-window update anchored on a locked safe order and concrete rollback guidance.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

## Dependency Notes

- `RN-102` before `RN-103`: The repo has no customer-facing safety check until dependency-fixture coverage lands, so the visible dashboard path should not expand before ordering regressions are catchable.
- `RN-102` before `RN-101`: Fixture backfill defines the safe ordering contract the later schema audit must validate against after the objective shift.
- `RN-101` before `RN-103`: The translated dashboard summary should only run behind the kill switch once schema stability is confirmed for the updated contract.
- `RN-103` before `RN-104`: Operators need documentation for the proven canary path and kill-switch behavior, not a speculative pre-canary checklist.
- `RN-104` before `RN-105`: Enterprise change-window communication depends on both correct ordering guarantees and operator rollback clarity.

## Primary Risk

If the dashboard or enterprise communication moves ahead of fixture-backed ordering checks and updated operator guidance, users can see and act on an incorrect release sequence during guarded change windows with no clear rollback path.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

Mitigations:
- Land dependency-graph fixture coverage before any broader dashboard rollout step.
- Require schema-drift audit completion before trusting the translated summary path.
- Update the operator runbook before external change-window messaging goes out.

## Assumption Ledger

- Objective priority [observed]: Current objective explicitly prioritizes safety and rollback confidence over dashboard speed.
- Legacy parser shim removal scope [to_verify]: Release notes mention auditing drift before shim removal, but the repo evidence does not show whether removing the shim is still in scope for this change window.
- Canary exit criteria [missing]: The provided evidence names a canary and kill switch but does not define the exact success or rollback thresholds for promoting the dashboard summary.

