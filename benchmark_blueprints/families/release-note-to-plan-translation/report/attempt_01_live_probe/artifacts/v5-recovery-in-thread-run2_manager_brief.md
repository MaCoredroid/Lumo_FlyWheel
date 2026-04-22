# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-order fixtures first

Restore missing dependency-graph fixture coverage so ordering regressions are caught before any schema or rollout decision is trusted. This is the smallest meaningful recovery milestone because the rollback was triggered by enabling the summary without updated fixtures.

Bounded deliverable: Fixture-backed dependency-order contract exists and is ready to gate downstream schema and rollout review work.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 2. RN-101 — Audit schema drift against the recovered contract

Run the translator schema drift audit only after the fixture backfill is in place, so legacy parser shim decisions are checked against the dependency-order contract instead of stale expectations.

Bounded deliverable: Schema drift is audited with fixture-backed evidence, and the legacy parser shim decision is grounded in the recovered ordering contract.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 3. RN-104 — Move runbook and checklist updates ahead of re-enable

Treat the original runbook timing as stale. The current objective and rollback follow-up both require folding rollback instructions and a structured launch checklist into the operator path before any summary restore decision, not after canary proof.

Bounded deliverable: Rollback-aware runbook and structured launch checklist are updated to reflect the verified dependency order and safe recovery path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `release_context/current_objective.md`
- `incident_context/rollback_incident.md`

### 4. RN-105 — Re-approve rollout after rollback review closes

Use re-approval as the explicit gate after fixture recovery, schema audit, and rollback-aware runbook/checklist updates are complete. The prior rollout-first plan was invalidated, so approval must happen on the corrected order.

Bounded deliverable: Rollback review is closed with an approval decision tied to recovered fixtures, audited schema, and updated operational guidance.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `incident_context/rollback_incident.md`

### 5. RN-103 — Re-enable the dashboard summary behind the kill switch

Restore the translated release-plan summary only after re-approval. The release note that frames this as merely a post-schema step is incomplete after the incident; the user-visible dashboard should stay off until the recovery chain is finished.

Bounded deliverable: A kill-switch-gated canary re-enable is allowed only after recovery prerequisites and formal re-approval are complete.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `release_context/current_objective.md`
- `incident_context/rollback_incident.md`

## Dependency Notes

- `RN-102` before `RN-101`: The schema audit must run against recovered dependency-order fixtures so drift decisions do not repeat the stale rollout-first failure.
- `RN-101` before `RN-104`: Rollback instructions and the launch checklist need the verified schema and ordering contract, otherwise the operator path documents the wrong dependency behavior.
- `RN-104` before `RN-105`: The dependency map says re-approval depends on an updated rollback and runbook path, so approval cannot close before the checklist and rollback guidance exist.
- `RN-105` before `RN-103`: The current objective explicitly blocks restoring the summary until rollback review finishes and the recovery path is approved.

## Primary Risk

If the dashboard summary is re-enabled before fixture recovery, schema audit, and rollback-aware runbook updates are complete, users will again see a misordered execution plan on a still-visible dashboard and support escalation will recur.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`
- `release_context/current_objective.md`

Mitigations:
- Gate all downstream work on fixture-backed dependency-order coverage before schema or rollout decisions.
- Require rollback-aware runbook and launch checklist updates before re-approval and any canary restore.
- Keep the summary behind the kill switch until the rollback review formally re-approves the corrected sequence.

## Assumption Ledger

- Rollback priority [observed]: Current objective explicitly prioritizes safe recovery over rollout velocity.
- Dashboard exposure [observed]: Repo state says the release dashboard remains user-visible, so ordering mistakes are externally visible.
- Canary exit criteria ownership [missing]: The repo evidence does not identify who owns the kill-switch canary exit decision or the exact success criteria for re-enable.
- Legacy parser shim disposition [to_verify]: Release notes require a schema drift audit before removing the shim, but the evidence set does not confirm whether removal is still in scope for this recovery window.

