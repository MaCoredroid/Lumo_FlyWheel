# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Live standalone flag with full runtime control of the shadow preview path. No blockers.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference it
- rationale: Alias that normalizes to ENABLE_SHADOW_PREVIEW. Should be deprecated after legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: operator notes may reference it, reporting systems may log it
- rationale: Dead flag with no runtime effect. Remove after operators and reporting systems stop referencing it.

