# Release Plan Brief

- Variant: `v4-multi-corpus-objective`
- First milestone: `rn-102-fixture-backfill`

## Ordered Plan

### 1. rn-102-fixture-backfill — Backfill dependency-order fixtures and safety checks

Land the dependency-graph fixtures that catch ordering regressions so the team has a concrete safety gate before any further rollout work.

Bounded deliverable: Fixture-backed ordering coverage is added to the plan contract and the missing customer-facing safety gap is closed.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 2. rn-104-operator-runbook — Update the operator runbook and launch checklist for rollback safety

Refresh the operator-facing rollback and launch guidance immediately after fixture coverage lands, because the current objective now prioritizes safety and rollback confidence over dashboard speed.

Bounded deliverable: Operators have an updated checklist and runbook that reflect the safe sequencing and rollback path for guarded change windows.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `release_context/current_objective.md`
- `repo_inventory/dependency_map.md`

### 3. rn-101-schema-audit — Audit schema drift before touching the legacy parser shim

Use the new fixture coverage and operator path to confirm the translator schema is stable before any legacy parser shim removal decision is made.

Bounded deliverable: A documented schema-drift decision is made with the shim retained or removed based on fixture-backed evidence rather than stale ordering assumptions.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/repo_state.md`

### 4. rn-103-dashboard-canary — Canary the translated dashboard summary behind a kill switch

Enable the translated release-plan dashboard summary only after the fixture, operator, and schema guardrails are in place, and keep it behind a kill switch while validating user-visible ordering.

Bounded deliverable: A kill-switched canary proves the summary path without exposing full rollout blast radius.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`
- `release_context/current_objective.md`

### 5. rn-105-enterprise-update — Send the enterprise change-window update once the order is locked

Communicate the corrected rollout order to enterprise customers only after the safe sequence is locked by fixtures, operator guidance, and canary evidence.

Bounded deliverable: Enterprise change-window communication goes out with a validated ordering story and rollback posture.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`
- `release_context/current_objective.md`

## Dependency Notes

- `rn-102-fixture-backfill` before `rn-104-operator-runbook`: The runbook should be updated only after ordering-regression coverage exists, otherwise the checklist documents an unverified safety path.
- `rn-102-fixture-backfill` before `rn-101-schema-audit`: Fixture coverage reduces ambiguity around whether schema changes alter release ordering, making the schema audit decision evidence-backed.
- `rn-104-operator-runbook` before `rn-103-dashboard-canary`: Because rollback confidence is now the primary objective, operator guidance must be ready before exposing even a kill-switched dashboard canary.
- `rn-101-schema-audit` before `rn-103-dashboard-canary`: The dashboard summary depends on a stable translator schema and an explicit decision about the legacy parser shim.
- `rn-103-dashboard-canary` before `rn-105-enterprise-update`: Enterprise change-window communication should cite proven canary behavior, not an unexercised rollout order.

## Primary Risk

If dashboard or enterprise rollout steps happen before fixture-backed ordering checks and rollback guidance, customers can see the wrong release-plan order during guarded change windows and operators will not have a trusted recovery path.

Evidence:
- `repo_inventory/repo_state.md`
- `release_context/current_objective.md`
- `repo_inventory/test_inventory.md`

Mitigations:
- Make dependency-fixture backfill the first milestone so ordering regressions are caught before rollout work resumes.
- Update the operator runbook before canary exposure so rollback and launch handling reflect the new safety-first objective.
- Keep the dashboard summary behind a kill switch until fixture, schema, and operator prerequisites are complete.

## Assumption Ledger

- Objective priority [observed]: The current objective explicitly shifted from dashboard speed to safety and rollback confidence.
- Schema regression coverage [observed]: Visible evidence says schema drift tests already exist, but dependency-order fixtures are still missing.
- Canary exit criteria [missing]: The scenario does not specify the exact success criteria or duration required to declare the kill-switched dashboard canary proven.

