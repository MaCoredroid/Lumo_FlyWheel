# SQLAlchemy 2 Cutover

## Changes Applied

- Repository layer now uses SQLAlchemy 2.x `select()` API instead of legacy `.query()`
- All transaction boundaries are explicit at the API/worker/admin_cli level
- Repository helpers no longer commit internally; callers manage transactions
- Proper rollback behavior on failures and dry-run operations

## Migration Notes

- API handlers use `session.begin()` context for explicit transactions
- Worker functions wrap operations in session context with single commit at end
- Admin CLI dry-run operations use `session.rollback()` to discard changes
- No legacy compatibility shim needed; full 2.x semantics applied
