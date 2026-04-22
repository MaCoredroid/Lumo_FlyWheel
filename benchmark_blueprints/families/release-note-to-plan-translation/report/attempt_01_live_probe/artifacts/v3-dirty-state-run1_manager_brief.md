# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and freeze the safe contract

Start by reconciling the frozen notes with the current repo state, because the in-progress draft incorrectly assumes the translator schema is already frozen. The milestone is a schema decision with an explicit keep-or-remove stance on the legacy parser shim, not a launch.

Bounded deliverable: A confirmed schema-drift audit and explicit translator contract decision that keeps the legacy shim in place until drift is resolved.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`
- `repo_inventory/dependency_map.md`

### 2. RN-102 — Backfill dependency-order fixtures and rollback coverage

Once the schema contract is stable enough to target, add the missing dependency-graph fixtures and rollback coverage so ordering regressions are caught before any user-visible summary path is enabled.

Bounded deliverable: Regression protection that exercises dependency reordering and rollback behavior for the translated plan contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 3. RN-103 — Canary the dashboard summary behind a kill switch

Only after schema and fixture guardrails are in place should the translated dashboard summary be enabled in a kill-switched canary, because the dashboard remains user-visible and is the first place operators would see bad ordering.

Bounded deliverable: A kill-switched canary of the translated release-plan dashboard summary with the safe path proven against the updated contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

### 4. RN-104 — Update the operator runbook after the canary proves stable

The runbook and launch checklist should trail the canary so operators are documented against the path that actually survived the schema and rollback checks, not the stale assumptions from earlier notes.

Bounded deliverable: Operator-facing runbook and checklist updates that match the proven canary behavior and kill-switch posture.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/in_progress_patch/README.md`

### 5. RN-105 — Review the abandoned rollout draft only as optional cleanup

The transferred-team draft should not drive the sequence. It depends on the same unresolved schema and fixture work as the dashboard summary, so it should be reviewed last to decide whether to finish it or discard it.

Bounded deliverable: A keep-or-drop decision for the abandoned rollout draft after the safe user path is already clear.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/in_progress_patch/README.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture and rollback coverage need a stable translator contract target; otherwise the tests would lock in the same false frozen-schema assumption as the abandoned draft.
- `RN-102` before `RN-103`: The dashboard is already user-visible, so kill-switch canary work must wait until dependency reordering and rollback regressions are actually caught.
- `RN-103` before `RN-104`: Operator guidance should describe the canary behavior that proved safe in practice, not the stale sequence implied by the frozen notes alone.
- `RN-101` before `RN-105`: The draft shares the unresolved schema dependency from the summary path, so reviewing it before the schema audit would anchor on invalid assumptions.
- `RN-102` before `RN-105`: The draft also lacks rollback coverage, making it optional cleanup only after the safe dependency and rollback path is protected.

## Primary Risk

If the dashboard summary or the abandoned rollout draft is pushed ahead of the schema audit and fixture backfill, operators will see a user-visible release-plan summary built on the wrong dependency order and without rollback protection.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Freeze the translator contract first and keep the legacy parser shim until schema drift is explicitly resolved.
- Add dependency-order fixtures and rollback coverage before enabling any translated summary path in canary.
- Treat the transferred-team rollout draft as optional cleanup instead of a prerequisite for the safe user path.

## Assumption Ledger

- Release context override [missing]: No release_context/ directory is present, so there is no newer operating objective overriding the frozen April notes.
- Incident-driven reorder [missing]: No incident_context/ directory is present, so there is no recorded rollback incident changing the sequence beyond the repo evidence.
- User-visible dashboard status [observed]: repo_inventory/repo_state.md explicitly says the release dashboard is still user-visible, so summary rollout errors would be externally visible.

