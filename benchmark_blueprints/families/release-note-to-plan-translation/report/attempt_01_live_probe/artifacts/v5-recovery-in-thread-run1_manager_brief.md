# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-ordering fixtures

Restore missing dependency-fixture coverage so ordering regressions are caught before any user-visible path is reconsidered.

Bounded deliverable: A dependency-graph fixture set and coverage pass that proves the translator rejects or flags misordered execution plans.

Evidence:
- `incident_context/rollback_incident.md`
- `repo_inventory/test_inventory.md`
- `release_context/current_objective.md`

### 2. RN-101 — Audit translator schema drift after fixtures are in place

Use the restored ordering fixtures to audit schema drift and decide whether the legacy parser shim can be retired safely.

Bounded deliverable: A schema-audit result tied to passing ordering fixtures, with any shim-retention decision documented.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `incident_context/rollback_incident.md`
- `repo_inventory/dependency_map.md`

### 3. RN-104 — Update the runbook and rollback checklist

Fold the rollback lessons into the operator runbook and produce the structured launch checklist that is currently missing.

Bounded deliverable: An updated runbook and launch checklist that describe rollback handling and the recovery gating sequence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`

### 4. RN-105 — Complete re-approval review for recovery

Run the formal recovery review only after fixtures, schema audit, and runbook updates are complete.

Bounded deliverable: A re-approval decision package showing recovery prerequisites are satisfied and the old rollout-first plan is invalidated.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 5. RN-103 — Re-enable the dashboard summary behind the kill switch

Only after recovery review passes, restore the translated release-plan dashboard summary with the kill switch still available.

Bounded deliverable: A controlled re-enable decision for the summary path, gated by the kill switch and backed by the recovery checklist.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-102` before `RN-101`: The rollback incident explicitly requires fixture backfill before the schema audit so the audit runs against the corrected ordering contract.
- `RN-101` before `RN-104`: The runbook should capture the validated schema and recovery behavior rather than documenting a stale pre-audit flow.
- `RN-104` before `RN-105`: The re-approval review depends on an updated rollback and launch path, not just code-facing fixes.
- `RN-105` before `RN-103`: The summary remains last because the prior user-visible incident came from enabling it before the recovery prerequisites and review were complete.

## Primary Risk

If the summary is re-enabled before fixture coverage, schema verification, and rollback-ready runbook updates are complete, users can again see a misordered execution plan in the release dashboard.

Evidence:
- `incident_context/rollback_incident.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Keep RN-102 as the first milestone so ordering regressions are caught before any rollout decision.
- Block RN-103 on completion of RN-101, RN-104, and RN-105 rather than treating schema stability alone as sufficient.
- Use the updated runbook and launch checklist to make rollback gating explicit for operators before re-enable.

## Assumption Ledger

- Dashboard exposure [observed]: repo_inventory/repo_state.md states the release dashboard is still user-visible, so any sequencing mistake remains customer-facing.
- Recovery ordering [observed]: incident_context/rollback_incident.md defines the corrected order as fixture backfill, then schema audit, then runbook and rollback instructions, then any re-enable decision.
- Kill-switch state while recovery is in progress [missing]: The evidence does not say whether the existing kill switch already blocks the bad summary path today, so the plan assumes re-enable must remain the final explicit decision.

