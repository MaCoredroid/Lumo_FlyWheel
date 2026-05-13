# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with direct runtime control. Essential for preview feature operation.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility. Should be deprecated once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: reporting systems using legacy_force_label
- rationale: Dead flag with no runtime effect. Remove after reporting migration.

