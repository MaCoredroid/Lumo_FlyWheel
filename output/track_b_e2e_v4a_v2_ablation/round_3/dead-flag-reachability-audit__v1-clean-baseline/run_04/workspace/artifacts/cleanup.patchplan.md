# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling preview mode. Essential for runtime branching.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility. Can be deprecated once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: Dead flag with no runtime effect. Remove parser code and update documentation to prevent operator confusion.

