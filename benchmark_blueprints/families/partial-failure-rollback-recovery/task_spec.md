# Partial-Failure Rollback Recovery

## Task Identity
- `task_id`: `partial-failure-rollback-recovery/rollback-state-repair`
- `family_id`: `partial-failure-rollback-recovery`
- `scenario_type`: `stateful_tool_execution`

## Task Prompt
Recover a local system after a partially failed maintenance action left state split across a manifest, a work queue, and a checkpoint ledger. Repair the rollback path so failed runs restore consistency without deleting evidence needed for later retry or audit.

## Workspace Bundle
- `maintenance/rollback.py`
- `maintenance/ledger.py`
- `fixtures/failed_runs/`
- `tests/test_rollback.py`, `tests/test_ledger.py`
- `docs/rollback_ops.md`

## Seeded Recovery Drift
- A naive rollback deletes evidence needed for audit.
- A second naive fix preserves evidence but leaves the queue inconsistent.
- One fixture represents a retryable failure and must not be treated as terminal cleanup.

## Required Surfaces
- `shell`
- `apply_patch`
- rollback reasoning
- terminal tests

## Expected Deliverables
- Rollback repair patch.
- Updated ops note.
- Validation evidence for retryable and terminal failure states.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_rollback.py tests/test_ledger.py`
- Hidden checks:
  - State consistency is restored.
  - Audit or retry evidence is preserved correctly.
  - Retryable failures are not over-cleaned.

## Quality Gate
- Target naive score: `20/100`.
