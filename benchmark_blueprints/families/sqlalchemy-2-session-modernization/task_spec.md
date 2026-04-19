# `sqlalchemy-2-session-modernization` Task Spec

## Task Prompt
Modernize the `ledger-sync` service from legacy SQLAlchemy `session.query(...)` patterns and implicit transaction behavior to SQLAlchemy 2.0 `select(...)` access with explicit session boundaries. The fix must hold across the API write path, the retrying worker, and the admin CLI dry-run and batch commands. Update the deploy note so operators understand the transaction-boundary change.

## Workspace Bundle
- `app/api.py`: HTTP write path for ledger entries.
- `app/repository.py`: shared query helpers and row loaders.
- `app/worker.py`: retryable settlement worker.
- `app/admin_cli.py`: dry-run and batch reconciliation commands.
- `app/models.py`, `app/db.py`: engine and session setup.
- `seed/ledger_seed.sql`: seeded records for visible and hidden checks.
- `docs/deploy/sqlalchemy2-cutover.md`: rollout note template with stale guidance.
- `README.md`: nearby local edits in dirty-workspace variants.
- `tests/test_api.py`, `tests/test_worker.py`, `tests/test_admin_cli.py`: visible tests.

## Seeded Modernization Breakage
- The dependency bump moved the service onto SQLAlchemy 2.x semantics.
- `repository.py` still uses `session.query(...)`, `.count()`, and implicit session state assumptions.
- `worker.py` relies on partial writes surviving exception paths, which now creates incorrect retry behavior.
- `admin_cli.py` treats dry-run as a real write path in one batch branch.
- The deploy note still suggests a broad compatibility shim rather than an explicit cutover.

## Required Surfaces
- `shell`
- `apply_patch`
- `terminal_tests`
- SQL and seed-data inspection
- docs update

## Expected Deliverables
- Code patch across repository, API, worker, and CLI layers.
- Explicit transaction-boundary handling that preserves rollback correctness.
- Updated deploy note documenting rollout, rollback, and operator-visible behavior changes.
- No blanket compatibility shim that preserves legacy `session.query(...)` behavior.

## Grader Contract
- Phase 2 visible checks:
  - `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py`
- Hidden checks:
  - API write path uses SQLAlchemy 2.0-style selects and persistence without leaking global session state.
  - Worker retry path is idempotent after a forced exception and does not leak partial writes.
  - Admin CLI dry-run performs read-only planning and batch mode persists only when expected.
  - `count`, `exists`, and `one_or_none` semantics remain correct after migration.
- Trusted final-state checks:
  - No live `session.query(` remains in the service path.
  - No blanket helper-level commit or global-session reuse was introduced to “make tests pass”.
  - Database state after a forced failure proves rollback happened.
  - Deploy note names the exact transaction-boundary change and rollback procedure.
- Scoring intent:
  - Syntax-only migration work is capped at `0.10`.
  - Visible green on one entrypoint is capped below `0.30`.
  - Scores above `0.50` require cross-entrypoint hidden checks and rollback-state invariants.

## Red-Team Traps
- Mechanical `query -> select` rewrite that ignores worker retry semantics.
- “Commit everywhere” patch that makes happy paths pass but breaks rollback.
- Docs-only or test-only updates.
- Fake compatibility helper that wraps legacy access patterns.
- Reverting nearby seed or README edits in dirty-workspace variants.
- Pytest spoofing, xfail-all, or deleting the visible tests.

## Quality Gate
- GPT-5.4/high adversarial probe judgment before hardening: `under_30_for_naive_solver = maybe`.
- Probe found a shallow path through a mostly mechanical 2.0 syntax migration plus a plausible deploy note.
- Hardening applied in this spec:
  - Bound the task to three entrypoints sharing the same transactional model.
  - Added rollback-state inspection and retry idempotence to the hidden suite.
  - Explicitly capped syntax-only credit and banned blanket commit/global-session escape hatches.
  - Made dry-run non-persistence part of the contract.
- Actual run evidence: [benchmark_run.md](./benchmark_run.md) records an overperforming first attempt, then a hardened rerun scored at `20/100`.
- Final quality judgment after hardening: `under_30_for_naive_solver = yes`.
- Rationale: the obvious migration pattern should earn some progress, but without fixing rollback, retry, and dry-run semantics across all surfaces it should not clear the 30-point threshold.
