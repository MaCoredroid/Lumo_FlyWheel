# Release Plan Brief

- Variant: `v3-dirty-state`
- First milestone: `RN-101`

## Ordered Plan

### 1. RN-101 — Audit translator schema drift and freeze the contract

Confirm the current translator schema and legacy shim boundaries before removing any compatibility path or building downstream rollout logic.

Bounded deliverable: A verified schema audit with the current drift documented and a stable contract baseline for follow-on work.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/in_progress_patch/README.md`

### 2. RN-102 — Backfill dependency-order fixtures and rollback coverage

Add the dependency-graph fixtures and regression checks needed to catch hidden ordering regressions before any user-facing summary path is exercised.

Bounded deliverable: A fixture-backed contract check that protects dependency ordering and covers rollback behavior gaps called out in the repo inventory.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/test_inventory.md`

### 3. RN-103 — Enable the dashboard summary behind a kill switch

Only after the schema and fixture guardrails are stable, expose the translated release-plan dashboard summary behind a reversible kill switch.

Bounded deliverable: A kill-switched dashboard summary path that can be enabled safely without making unstable ordering logic the default user experience.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 4. RN-104 — Prove the summary path in canary and update operator docs

After the guarded summary path works in canary, update the operator runbook and launch checklist to match the validated rollout sequence.

Bounded deliverable: A canary-proven summary path plus operator documentation that reflects the actual guarded launch steps.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/repo_state.md`

### 5. RN-105 — Decide whether to discard or finish the abandoned rollout draft

Treat the transferred engineer's draft as optional cleanup only after the safe production path is clear, because it currently rests on invalid assumptions.

Bounded deliverable: A disposition decision on the abandoned draft with either explicit discard rationale or a narrowed follow-up scope.

Evidence:
- `release_notes/release_notes_2026_04.md`
- `repo_inventory/dependency_map.md`

## Dependency Notes

- `RN-101` before `RN-102`: Fixture expectations should be written against the audited translator contract, otherwise the regression suite may codify the wrong schema assumptions.
- `RN-102` before `RN-103`: The dashboard summary is already user-visible, so dependency-order guardrails need to exist before the new translated summary path is exposed, even behind a kill switch.
- `RN-103` before `RN-104`: Operators should update the runbook and launch checklist only after the guarded summary path has been proven in canary.
- `RN-101` before `RN-105`: The in-progress draft assumes a frozen schema today; reviewing its fate before the schema audit would anchor the decision on a known-false premise.
- `RN-102` before `RN-105`: The dependency map says the abandoned draft shares the unresolved schema and fixture prerequisites of the safe dashboard path, so it should remain optional cleanup after those blockers are cleared.

## Primary Risk

If dashboard or rollout-draft work is started before the schema audit and dependency-order fixtures are in place, users can see a release dashboard summary that reflects the wrong execution order, and operators may adopt a launch checklist that cannot survive rollback.

Evidence:
- `repo_inventory/repo_state.md`
- `repo_inventory/test_inventory.md`
- `repo_inventory/in_progress_patch/README.md`

Mitigations:
- Complete RN-101 first to freeze the translator contract before removing the legacy shim or extending rollout logic.
- Complete RN-102 next so dependency-order and rollback regressions are caught before any user-facing summary path is exercised.
- Keep RN-103 behind a kill switch and defer RN-104 operator documentation until the canary result proves the guarded path.

## Assumption Ledger

- release_context [missing]: No release_context directory is present, so the plan assumes the frozen April notes still reflect the current operating objective.
- incident_context [missing]: No incident_context directory is present, so there is no rollback-specific replanning input beyond the repo inventory notes.
- canary_success_criteria [to_verify]: The repo inventory indicates canary proof is required before runbook updates, but it does not define the exact acceptance threshold.

