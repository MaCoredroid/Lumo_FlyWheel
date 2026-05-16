# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling preview rollout. Essential for runtime branching logic.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Live alias for backward compatibility. Deprecate after legacy manifests migrate to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: operator notes, reporting dashboards
- rationale: Dead flag with no runtime effect. Remove after migrating operator notes and telemetry dashboards to stop referencing it.

