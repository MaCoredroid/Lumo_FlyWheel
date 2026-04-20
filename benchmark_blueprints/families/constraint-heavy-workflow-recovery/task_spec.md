# Constraint-Heavy Workflow Recovery

## Task Identity
- `task_id`: `constraint-heavy-workflow-recovery/recovery-plan-and-fix`
- `family_id`: `constraint-heavy-workflow-recovery`
- `scenario_type`: `stateful_tool_execution`

## Task Prompt
Recover a partially completed local workflow that must satisfy multiple operational constraints at once: ordering, authorization, data-retention, and reporting. Repair the workflow so it completes legally and safely after a mid-run failure, then explain the recovery boundaries.

## Workspace Bundle
- `workflow/engine.py`
- `workflow/policies.py`
- `fixtures/partial_runs/`
- `tests/test_constraints.py`, `tests/test_recovery.py`
- `docs/recovery_ops.md`

## Seeded Constraint Drift
- One visible fix path satisfies ordering but violates retention.
- A second path restores state but loses required reporting.
- The failing workflow snapshot already contains one irreversible step.

## Required Surfaces
- `shell`
- `apply_patch`
- constraint reasoning
- terminal tests

## Expected Deliverables
- Workflow recovery patch.
- Constraint-aware recovery note.
- Validation evidence for hidden constraint combinations.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_constraints.py tests/test_recovery.py`
- Hidden checks:
  - Ordering, authorization, retention, and reporting all hold simultaneously.
  - Mid-run recovery does not replay irreversible steps incorrectly.
  - Recovery note matches the implemented boundaries.

## Quality Gate
- Target naive score: `20/100`.
