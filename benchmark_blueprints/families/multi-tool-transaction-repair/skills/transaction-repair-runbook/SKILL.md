# Transaction Repair Runbook

Use this skill when a task requires repairing a local multi-service transaction under partial-failure conditions.

## Workflow
1. Trace the state mutation order across all touched services.
2. Identify the first irreversible side effect.
3. Repair rollback and retry behavior before changing documentation.
4. Validate both happy-path and partial-failure states.

## Anti-Patterns
- Fixing only the visible happy path.
- Accepting duplicate notifications as harmless.
- Updating the operator note without restoring transaction safety.

## Done Signal
- The transaction is atomic again and partial failures leave no duplicated side effects.
