# Release Plan Brief

- Variant: `v5-recovery-in-thread`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-order fixtures

Make the first milestone the missing dependency-graph fixture backfill so ordering regressions are caught before any more rollout work proceeds. This is the rollback-corrected starting point, not the rollout-first order implied by the frozen notes.

Bounded deliverable: A bounded fixture set and plan-contract coverage update that fails when execution steps are misordered.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 2. RN-101 — Audit schema drift and parser-shim exit

Audit translator schema drift only after the fixture backfill is in place, so the schema decision is tested against the ordering contract. This keeps the legacy parser-shim decision tied to verified behavior instead of stale assumptions.

Bounded deliverable: A schema-audit result with an explicit keep-or-remove decision for the legacy parser shim against updated fixtures.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `incident_context/rollback_incident.md`
- `repo_inventory/dependency_map.md`

### 3. RN-104 — Update rollback runbook and launch checklist

Fold the rollback notes into the operator runbook and create the structured launch checklist before any re-enable decision. The operating objective explicitly prioritizes safe recovery over rollout velocity, and the current rollback path only exists in docs.

Bounded deliverable: An updated runbook and launch checklist that encode rollback handling, gating, and operator steps for the summary path.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `incident_context/rollback_incident.md`

### 4. RN-105 — Re-approve the summary rollout gate

Run the rollback review and re-approval only after fixtures, schema audit, and rollback instructions exist. This explicitly supersedes the invalidated rollout-first plan and turns the review into a readiness gate rather than a status meeting.

Bounded deliverable: A recorded re-approval decision with kill-switch preconditions and explicit dependency signoff.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`

### 5. RN-103 — Canary re-enable the dashboard summary behind a kill switch

Re-enable the translated release-plan dashboard summary only as a controlled canary after the review gate passes. The frozen note is still valid as a destination, but its original position in the sequence is stale because the incident and current objective moved rollout behind recovery work.

Bounded deliverable: A kill-switched canary of the translated summary path with rollback instructions linked from the operator flow.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `incident_context/rollback_incident.md`

## Dependency Notes

- `RN-102` before `RN-101`: The rollback follow-up explicitly requires fixture backfill first so schema decisions are checked against ordering regressions.
- `RN-101` before `RN-104`: The runbook and launch checklist need the post-audit schema and parser-shim state, not a draft contract.
- `RN-104` before `RN-105`: The dependency map says re-approval depends on an updated rollback and runbook path.
- `RN-102` before `RN-105`: The re-approval review is blocked until dependency-fixture coverage exists for ordering regressions.
- `RN-105` before `RN-103`: Any summary re-enable must follow rollback review signoff and kill-switch approval to avoid repeating the incident.

## Primary Risk

If the summary is re-enabled before fixture backfill, schema audit, and rollback instructions are complete, users will again see a misordered execution plan on the still-visible dashboard and support will face another fast rollback escalation.

Evidence:
- `incident_context/rollback_incident.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

Mitigations:
- Keep recovery work ahead of rollout work by backfilling dependency fixtures before the schema audit and any launch decision.
- Require the updated rollback runbook and structured launch checklist before the re-approval gate can pass.
- Limit re-enable scope to a kill-switched canary instead of a broad summary restoration.

## Assumption Ledger

- Frozen rollout order is stale [observed]: The rollback incident and objective shift invalidate the earlier rollout-first sequencing implied by the frozen notes.
- Current summary exposure state [missing]: Repo state says the dashboard remains user-visible, but the inputs do not confirm whether the translated summary path is fully dark or partially exposed today.
- Canary exit owner and thresholds [to_verify]: The inputs require a re-approval gate and kill switch, but they do not name the owner or concrete canary success thresholds.

