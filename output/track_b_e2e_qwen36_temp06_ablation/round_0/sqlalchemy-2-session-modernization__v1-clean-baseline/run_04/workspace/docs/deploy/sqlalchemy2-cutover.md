# SQLAlchemy 2 Cutover

Repository access now uses SQLAlchemy 2.x `select()` syntax instead of the
legacy `session.query()` API.  The `session.commit()` calls have been removed
from repository helpers — transaction ownership belongs to the caller.

Each entry point (API, worker, admin CLI) manages its own explicit transaction
boundaries with `try/except` + `session.rollback()` so that failures do not
leave partial state on disk.  Dry-run paths in the admin CLI roll back instead
of committing.
