# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `RN-102`

## Ordered Plan

### 1. RN-102 — Backfill dependency-graph fixtures before any rollout expansion

Land the missing dependency-fixture coverage so ordering regressions are caught by the plan contract instead of surfacing during guarded change windows.

Bounded deliverable: Fixture-backed ordering tests exist for the translated release-plan path and are treated as the gate for downstream rollout work.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 2. RN-101 — Audit translator schema drift while retaining the legacy parser shim

Use the now-hardened fixture coverage to verify schema stability and only then decide whether the legacy parser shim can be removed safely.

Bounded deliverable: A documented schema-drift audit with an explicit keep-or-remove decision for the legacy shim, backed by green contract coverage.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Canary the translated dashboard summary behind the kill switch

Roll the translated summary path out cautiously only after ordering guarantees and schema stability are established, using the kill switch as the rollback lever.

Bounded deliverable: A kill-switch-protected canary of the translated dashboard summary with clear rollback handling.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/dependency_map.md`

### 4. RN-104 — Update the operator runbook and launch checklist after the canary proves out

Refresh operator guidance only once the canary demonstrates the safe plan order and the rollback path are real, so operators do not codify stale sequencing.

Bounded deliverable: Runbook and launch checklist updates that match the proven canary behavior and rollback procedure.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`

### 5. RN-105 — Send the enterprise change-window update once the safe order is locked

Communicate to enterprise customers only after the implementation order, operator playbook, and rollback handling are fixed so the update does not repeat the escalated sequencing error.

Bounded deliverable: An enterprise change-window communication based on the locked safe order and validated operator process.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-102` before `RN-101`: The fixture backfill supplies the missing ordering-regression gate, so the schema audit and shim decision are made against the safety objective instead of drift checks alone.
- `RN-101` before `RN-103`: The translated dashboard summary should not enter canary until schema stability has been audited and the shim-removal decision is explicit.
- `RN-103` before `RN-104`: Operator instructions need the proven canary path and rollback behavior, otherwise the checklist can memorialize an unverified sequence.
- `RN-104` before `RN-105`: Enterprise communication should follow the validated operator runbook so external guidance matches the actual guarded change-window procedure.

## Primary Risk

If rollout, runbook, or enterprise communication moves ahead before the ordering-fixture gate is in place, users will keep seeing an apparently valid dashboard and operator process that encode the wrong release-plan order during guarded change windows.

Evidence:
- `repo_inventory/repo_state.md`
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Make RN-102 the first milestone and treat fixture-backed ordering coverage as the gate for all downstream work.
- Keep RN-103 behind the kill switch until RN-101 confirms schema stability and the shim decision is explicit.
- Delay RN-104 and RN-105 until the canary proves the safe order and rollback path.

## Assumption Ledger

- Legacy parser shim behavior under dependency-fixture coverage [observed]: The shim remains available today, so the immediate plan can audit drift without coupling rank 1 to shim removal.
- Canary exit criteria for the translated dashboard summary [missing]: The repo evidence does not define what constitutes a successful canary, so the plan assumes product and operations will need to specify the acceptance threshold before broad rollout.

