# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag for shadow preview rollout. Must remain for operational control.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias for ENABLE_SHADOW_PREVIEW. Can be removed after legacy deploy manifests migrate to use primary flag directly.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: telemetry consumers, operator notes
- rationale: Dead flag with no runtime effect. Remove after telemetry consumers and operator documentation updated to not expect this flag.

