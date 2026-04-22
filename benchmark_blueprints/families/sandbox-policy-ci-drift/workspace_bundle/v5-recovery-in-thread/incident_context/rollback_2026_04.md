
# Rollback 2026-04

The previous rename-only hotfix removed `workspace-write` input support.
Stored examples still used that spelling, so the patch was rolled back
after breaking local dry runs. Preserve compatibility on input while
keeping emitted preview and workflow tokens canonical.
