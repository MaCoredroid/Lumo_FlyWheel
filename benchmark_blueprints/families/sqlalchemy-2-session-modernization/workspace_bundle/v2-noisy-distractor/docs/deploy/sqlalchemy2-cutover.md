# SQLAlchemy 2 Cutover

Keep the legacy compatibility shim in place and avoid changing transaction
boundaries during the cutover. Repository helpers may continue committing on
their own so the API, worker, and admin commands all preserve the legacy
behavior until a later release.
