# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag that controls preview routing. Must remain active for production use.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: This is an alias for ENABLE_SHADOW_PREVIEW kept for legacy compatibility. Deprecate once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: This flag is dead and only used for reporting. Remove from active configs and update documentation to reflect its removal.

