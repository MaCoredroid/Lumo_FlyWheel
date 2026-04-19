# Transaction Boundary Audit

Use this skill when a task modernizes ORM access and transaction behavior across multiple entrypoints.

## Workflow
1. Inspect write paths before doing syntax migration.
2. Separate API, worker, and CLI transaction boundaries.
3. Verify failure paths, retry behavior, and dry-run behavior.
4. Treat docs as proof of the landed transaction model, not as a substitute.

## Anti-Patterns
- Mechanical `query -> select` rewrites.
- Blanket `commit()` calls to make tests pass.
- Shared global session state across entrypoints.

## Done Signal
- Visible tests pass and the transaction model is explicit per surface.
