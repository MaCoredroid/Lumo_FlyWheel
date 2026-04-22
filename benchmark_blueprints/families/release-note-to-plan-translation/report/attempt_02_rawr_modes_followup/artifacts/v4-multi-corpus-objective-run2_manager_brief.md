# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-order fixtures and lock the safe plan contract

Add the missing dependency-graph fixtures so ordering regressions are caught before any user-visible rollout or enterprise communication proceeds.

Bounded deliverable: Dependency-fixture coverage is present for translated release-plan ordering and serves as the first customer-facing safety gate.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 2. RN-101 — Audit translator schema drift and decide legacy parser shim disposition

Use the new fixture coverage to verify schema stability and make an explicit keep-or-remove decision on the legacy parser shim before downstream rollout work.

Bounded deliverable: A schema-stability audit with a documented shim decision that downstream rollout steps can rely on.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the translated dashboard summary behind the kill switch

Only after ordering guardrails and schema stability are in place, prove the translated summary path in a reversible canary while keeping rollout gated behind the kill switch.

Bounded deliverable: A kill-switched canary result that demonstrates the translated dashboard summary can be rolled back safely.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Update the operator runbook and launch checklist from the proven canary path

Capture the validated execution order, rollback posture, and launch checklist after the canary proves the summary path so operators are not working from stale rollout assumptions.

Bounded deliverable: Runbook and checklist reflect the proven safe order and rollback steps for guarded change windows.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`

### 5. RN-105 — Send the enterprise change-window update after the order is locked

Communicate externally only once the safe plan order, rollback posture, and operator guidance are all locked so the enterprise message reflects the actual guarded rollout path.

Bounded deliverable: Enterprise change-window update references a validated execution order instead of a stale or dashboard-first sequence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-102` before `RN-101`: The schema audit should run after dependency-order fixtures exist, because the current gap is the missing safety check for ordering regressions.
- `RN-101` before `RN-103`: The dashboard canary is only safe once schema stability is audited and the legacy shim decision is explicit.
- `RN-103` before `RN-104`: Operator guidance must reflect the proven canary path rather than the stale pre-escalation rollout assumptions.
- `RN-104` before `RN-105`: The enterprise change-window update should be sent only after operators have a validated launch and rollback checklist to execute.

## Primary Risk

If dashboard rollout or enterprise communication happens before dependency fixtures and updated operator guidance land, users can see incorrect release-plan ordering during guarded change windows with no reliable safety gate or rollback playbook.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Make dependency-fixture backfill the first milestone so ordering regressions are blocked before rollout.
- Keep dashboard exposure behind the kill switch until the schema audit and canary are complete.
- Update the operator runbook and checklist before sending the enterprise change-window notice.

## Assumption Ledger

- incident_context availability [missing]: No incident_context directory is present, so the plan assumes the current objective shift captures the relevant rollback lessons.
- dashboard visibility [observed]: Repo state explicitly says the release dashboard remains visible to users.
- change-window communication gate [observed]: Dependency map says enterprise change-window communication is blocked on correct ordering guarantees.

