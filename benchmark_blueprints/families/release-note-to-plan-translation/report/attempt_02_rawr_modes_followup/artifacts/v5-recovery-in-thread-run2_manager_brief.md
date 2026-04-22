# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-order fixtures

Restore the missing dependency-graph fixture coverage so ordering regressions are caught before any recovery work proceeds.

Bounded deliverable: Fixture backfill committed and used as the gating evidence for release-plan ordering checks.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 2. RN-101 — Audit schema drift against the recovered fixture baseline

Use the refreshed ordering fixtures to audit translator schema drift before any legacy-parser cleanup or downstream rollout decision.

Bounded deliverable: Schema-audit decision that confirms the translator output is stable enough for recovery planning.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `incident_context/rollback_incident.md`

### 3. RN-104 — Fold rollback handling into the operator runbook and launch checklist

Update the operator path with structured rollback and launch guidance before any re-enable decision, because the original post-canary timing is superseded by the recovery objective.

Bounded deliverable: Runbook plus structured launch checklist that documents rollback handling and safe recovery sequencing.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `release_notes/release_notes_2026_04.md`
- `incident_context/rollback_incident.md`

### 4. RN-105 — Complete rollback review and re-approve the rollout

Run the formal recovery review only after fixtures, schema, and runbook/checklist updates are complete so approval reflects the corrected launch path.

Bounded deliverable: Explicit re-approval decision tied to the recovered dependency, schema, and rollback evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

### 5. RN-103 — Re-enable the dashboard summary behind the kill switch

Only after re-approval, restore the translated release-plan summary in a controlled canary path behind the kill switch.

Bounded deliverable: Kill-switch-guarded summary canary enabled after recovery prerequisites are satisfied.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

## Dependency Notes

- `RN-102` before `RN-101`: The rollback incident explicitly attributes the user-visible failure to missing dependency fixtures, so schema audit must happen against the corrected ordering baseline.
- `RN-101` before `RN-104`: Rollback and launch instructions need the validated schema and ordering behavior, otherwise the operator guidance will encode stale recovery steps.
- `RN-104` before `RN-105`: The dependency map says re-approval requires an updated rollback/runbook path, not just technical fixes.
- `RN-105` before `RN-103`: The current objective forbids restoring the user-visible summary until the rollback review finishes and the safe recovery path is approved.

## Primary Risk

If the team re-enables the dashboard summary before fixtures, schema validation, and rollback instructions are in place, users will again see a misordered execution plan on the still-visible dashboard and support will escalate quickly.

Evidence:
- `repo_inventory/repo_state.md`
- `incident_context/rollback_incident.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Make the fixture backfill the first milestone and gate downstream steps on it.
- Require schema audit plus rollback/runbook checklist completion before re-approval.
- Keep summary re-enable behind the kill switch until the recovery review approves the corrected path.

## Assumption Ledger

- Canary success criteria [missing]: The notes require a canary-style recovery path, but the exact acceptance threshold for re-enabling the summary is not present in the workspace evidence.
- RN-104 timing [observed]: The original release note placed the runbook update after canary, but the current objective and rollback evidence supersede that ordering for recovery.

