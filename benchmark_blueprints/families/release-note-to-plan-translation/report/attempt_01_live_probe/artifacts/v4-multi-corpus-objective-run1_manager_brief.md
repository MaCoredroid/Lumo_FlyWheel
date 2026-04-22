# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill ordering fixtures as the safety gate

Make dependency-fixture coverage the first milestone because the current objective shifted to safety and rollback confidence, and the repo still lacks a customer-facing check that catches incorrect release-plan ordering.

Bounded deliverable: A dependency-fixture suite that fails on wrong ordering and restores the missing customer-facing safety check.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `release_context/current_objective.md`

### 2. RN-101 — Audit schema drift before any parser-shim decision

Once ordering regressions are covered, audit translator schema drift and explicitly hold any legacy parser shim removal until the plan contract is proven stable across both schema and dependency dimensions.

Bounded deliverable: A schema-drift audit outcome that confirms whether the legacy shim can stay, be narrowed, or be removed safely.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-104 — Update operator runbook and rollback checklist early

Move the runbook and launch-checklist work ahead of dashboard expansion, because the frozen note order predates the enterprise escalations and the updated objective now prioritizes operator clarity over dashboard speed.

Bounded deliverable: An operator-facing checklist that documents the safe plan order, rollback triggers, and what to do if ordering confidence degrades.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 4. RN-103 — Canary the dashboard summary behind the kill switch

Only after ordering fixtures, schema validation, and operator guidance are in place should the translated dashboard summary advance behind its kill switch, so the already-visible dashboard does not amplify a known sequencing hazard.

Bounded deliverable: A constrained canary of the summary path with the kill switch ready and operators trained on rollback handling.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

### 5. RN-105 — Send the enterprise change-window update last

Keep enterprise communication last, because the dependency map explicitly blocks the change-window message on correct ordering guarantees and the safe sequence is not locked until the earlier safeguards and canary are complete.

Bounded deliverable: A change-window update that reflects the locked safe order, current rollback posture, and proven rollout path.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

## Dependency Notes

- `RN-102` before `RN-101`: Ordering-fixture coverage closes the current safety gap first, so the later schema audit is evaluated against the full contract instead of a partial gate.
- `RN-101` before `RN-104`: Operators need the validated schema and shim posture before the runbook can describe a trustworthy rollback path.
- `RN-104` before `RN-103`: The objective shift makes operator clarity a prerequisite for dashboard canary work, even though the frozen notes placed runbook updates later.
- `RN-103` before `RN-105`: Enterprise change-window communication is blocked until the safe ordering guarantees and kill-switch rollout path are proven.

## Primary Risk

If dashboard or communication work moves ahead of fixture backfill and operator guidance, enterprise users can see or act on incorrect release-plan ordering during guarded change windows with no clear rollback response.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

Mitigations:
- Land dependency-fixture coverage before any broader rollout or external communication.
- Document the safe order and rollback procedure for operators before summary canary exposure expands.
- Use the dashboard kill switch as a containment step, not as the primary safety mechanism.

## Assumption Ledger

- Objective priority [observed]: The current objective explicitly changed from dashboard speed to safety and rollback confidence after enterprise escalations.
- Incident-specific rollback evidence [missing]: No incident_context directory is present, so the plan cannot cite a concrete prior rollback artifact beyond the current objective shift.
- Legacy parser shim blast radius [to_verify]: The frozen notes require a schema-drift audit before shim removal, but the inputs do not show whether downstream tooling still depends on the shim.

