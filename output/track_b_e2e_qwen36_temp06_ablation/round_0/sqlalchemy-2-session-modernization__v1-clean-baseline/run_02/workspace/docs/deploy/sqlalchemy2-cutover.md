# SQLAlchemy 2 Cutover

## Changes Applied

- **Repository layer** — migrated from legacy `session.query()` to SQLAlchemy 2.x `select()` statements. Removed internal `session.commit()` calls from repository helpers so callers control transaction boundaries.

- **API (`app/api.py`)** — the caller's transaction boundary (e.g., `session_factory.begin()`) now wraps both create and read operations without being broken by an internal commit.

- **Worker (`app/worker.py`)** — `settle_entry` no longer commits after marking "processing"; the entire operation (mark + settle) is a single transaction that rolls back on error.

- **Admin CLI (`app/admin_cli.py`)** — `plan_batch` dry-run now returns without committing, so status changes are rolled back when the session exits. `reconcile_batch` commits once after all entries are processed.

## Migration Notes

- Repository helpers are now passive — they mutate session state but do not commit. The caller is responsible for explicit `session.commit()` or rollback.
- All queries use `select()` with `session.execute()` and `scalar_one()` / `scalars()` instead of the legacy `Query` API.
