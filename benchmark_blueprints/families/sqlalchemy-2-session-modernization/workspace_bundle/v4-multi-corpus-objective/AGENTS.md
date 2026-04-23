You are modernizing a small `ledger-sync` service to SQLAlchemy 2.x semantics.

Work inside this workspace only. The intended fix is behavioral, not just
syntactic: repository access should use SQLAlchemy 2-style selects, and the API,
worker, and admin CLI should each have explicit transaction boundaries with
correct rollback behavior.

Allowed edits:
- `app/*.py`
- `docs/deploy/sqlalchemy2-cutover.md`

Do not modify tests, seed data, or any contextual note files. Preserve any file
that looks like surrounding project context unless the task prompt explicitly
requires changing it.

Validation surface:
- `pytest -q tests/test_api.py tests/test_worker.py tests/test_admin_cli.py`

Deliver a real transaction-boundary modernization. Do not solve this by adding
blanket helper-level commits, a global session, or a legacy compatibility shim.
This variant adds release-context files. Use them to decide what operators need
the cutover note to emphasize, especially around dry-run safety and rollback.
