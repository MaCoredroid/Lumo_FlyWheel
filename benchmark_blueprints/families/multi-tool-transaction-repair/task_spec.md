# Multi-Tool Transaction Repair

## Task Identity
- `task_id`: `multi-tool-transaction-repair/stateful-transaction-fix`
- `family_id`: `multi-tool-transaction-repair`
- `scenario_type`: `stateful_tool_execution`

## Task Prompt
Repair a broken multi-step transaction workflow in a frozen local service sandbox. The workflow touches an order store, a billing ledger, and a notification queue. Preserve atomicity, avoid duplicate side effects, and provide a short operator note explaining the repaired transaction boundaries.

## Workspace Bundle
- `services/orders/`, `services/billing/`, `services/notifications/`
- `fixtures/state/`: partially applied transaction snapshots.
- `tests/test_transaction_flow.py`, `tests/test_atomicity.py`
- `docs/transaction_ops.md`

## Seeded State Drift
- Visible tests cover the happy path but not partial rollback.
- One service retries too aggressively and can emit a duplicate side effect.
- The existing operator note describes the pre-retry behavior only.

## Required Surfaces
- `shell`
- `apply_patch`
- local multi-service reasoning
- terminal tests

## Expected Deliverables
- Code patch for transaction recovery.
- Updated operator note.
- Validation evidence showing no duplicate side effects.

## Grader Contract
- Visible checks:
  - `pytest -q tests/test_transaction_flow.py tests/test_atomicity.py`
- Hidden checks:
  - Partial-failure state rolls back safely.
  - Notification side effects are not duplicated.
  - Repair preserves atomicity across services.

## Quality Gate
- Target naive score: `20/100`.
