# SQLAlchemy 2 Cutover

Repository queries now use SQLAlchemy 2-style `select()` statements instead of
the legacy `session.query()` API.  Repository helpers no longer issue their own
`session.commit()` — transaction boundaries are managed at the call site.

The API, worker, and admin CLI each wrap their work in an explicit
`session_factory.begin()` block.  On success the block commits automatically;
on exception the session rolls back, so intermediate status changes are never
persisted.
