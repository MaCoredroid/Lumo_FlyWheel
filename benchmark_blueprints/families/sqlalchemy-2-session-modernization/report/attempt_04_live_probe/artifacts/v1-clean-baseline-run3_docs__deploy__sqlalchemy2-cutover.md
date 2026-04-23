# SQLAlchemy 2 Cutover

Adopt SQLAlchemy 2-style repository access and move transaction ownership to the
service boundary.

- Repository helpers should use `select(...)`-based reads and must not commit.
- API calls should run inside an explicit caller-owned transaction so related
  reads and writes stay in the same unit of work.
- Worker jobs should wrap each settlement flow in a single transaction so any
  exception rolls back intermediate status changes.
- Admin commands should decide whether to commit or roll back at the command
  boundary, including dry-run paths that intentionally discard staged changes.
