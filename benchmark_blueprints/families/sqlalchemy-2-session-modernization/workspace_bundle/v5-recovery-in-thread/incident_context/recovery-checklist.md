## Recovery Checklist

Before retrying the cutover:
- prove rollback leaves rows at `pending` after forced failures
- prove dry-run stays read-only
- document the rollback steps for operators
