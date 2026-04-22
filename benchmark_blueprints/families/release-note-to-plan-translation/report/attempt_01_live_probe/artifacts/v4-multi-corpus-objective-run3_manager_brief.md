# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `RN-102-fixture-guardrails`

## Ordered Plan

### 1. RN-102-fixture-guardrails — Backfill dependency fixtures as the first safety gate

Land dependency-graph fixture coverage against the current translator behavior so ordering regressions become visible before any customer-facing rollout expands.

Bounded deliverable: A fixture-backed ordering contract that closes the current safety-check gap without changing dashboard exposure yet.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/dependency_map.md`

### 2. RN-101-schema-audit — Audit schema drift while the legacy parser shim stays in place

Use the new fixtures to audit translator schema drift and confirm whether the legacy parser shim can be removed without reintroducing ordering regressions.

Bounded deliverable: A documented schema-stability decision with the shim either retained temporarily or cleared for later removal based on fixture evidence.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/repo_state.md`

### 3. RN-103-dashboard-canary — Canary the translated dashboard summary behind the kill switch

Only after fixture coverage and schema audit are in place, prove the translated summary path in canary with the kill switch available for fast rollback.

Bounded deliverable: A limited canary result showing the summary path is stable enough to expose without widening user-visible ordering mistakes.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/repo_state.md`

### 4. RN-104-runbook-checklist — Update the operator runbook and launch checklist for rollback clarity

Refresh operator guidance after canary evidence exists so guarded change-window handling reflects the safe order, rollback path, and kill-switch usage.

Bounded deliverable: An operator-facing runbook and checklist aligned to the proven order instead of the superseded dashboard-speed priority.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`

### 5. RN-105-enterprise-update — Send the enterprise change-window update after the safe order is locked

Communicate with enterprise customers only once the ordering guarantees, rollback guidance, and operator checklist are settled enough to support guarded windows.

Bounded deliverable: A customer-ready change-window update that reflects the validated sequence instead of an outdated rollout story.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/test_inventory.md`

## Dependency Notes

- `RN-102-fixture-guardrails` before `RN-101-schema-audit`: The repo inventory says there is no customer-facing safety check until fixture backfill lands, so schema decisions need that guardrail first.
- `RN-101-schema-audit` before `RN-103-dashboard-canary`: The release notes gate dashboard enablement on schema stability, and the current objective says safety now outranks dashboard speed.
- `RN-103-dashboard-canary` before `RN-104-runbook-checklist`: The operator runbook should describe the proven canary path, kill-switch behavior, and rollback flow instead of draft assumptions.
- `RN-104-runbook-checklist` before `RN-105-enterprise-update`: Enterprise communication is blocked on correct ordering guarantees and should include the final operator-safe procedure for guarded windows.

## Primary Risk

If dashboard rollout or enterprise communication happens before fixture-backed ordering proof and rollback-ready operator guidance, users will continue seeing incorrect release-plan order during guarded change windows and may execute dependent work in the wrong sequence.

Evidence:
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Make dependency-fixture backfill the first milestone so ordering regressions are caught before rollout widens.
- Keep dashboard exposure behind the existing kill switch until schema stability is audited against the new fixtures.
- Delay the enterprise change-window update until the runbook and checklist reflect the proven safe order and rollback path.

## Assumption Ledger

- Objective priority [observed]: The current objective explicitly shifted from dashboard speed to safety and rollback confidence after enterprise escalations.
- Legacy parser shim outcome [to_verify]: The notes require a schema-drift audit before removing the shim, but the inputs do not prove whether removal is safe yet.
- Canary exit criteria [missing]: The workspace does not specify quantitative canary success thresholds or exact rollback triggers for the dashboard summary path.

