# Rollout Order

The reusable workflow now emits an `artifact_manifest` output.
Operators must confirm that output before the staging smoke step,
because the dry-run path can stay green while the manifest contract is stale.
