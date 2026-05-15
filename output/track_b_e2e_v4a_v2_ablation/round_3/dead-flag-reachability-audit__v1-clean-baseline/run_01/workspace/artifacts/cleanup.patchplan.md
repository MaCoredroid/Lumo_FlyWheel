# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag - do not remove

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Legacy alias still needed for backward compatibility but should be deprecated in favor of ENABLE_SHADOW_PREVIEW

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: Parser symbol can be removed once reporting is migrated

