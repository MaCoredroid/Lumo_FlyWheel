# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live control for shadow preview routing. Essential for feature rollout operations.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference it
- rationale: Alias flag that should be deprecated after legacy manifests migrate to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator notes may reference it
- rationale: Remove from active config parsing but keep telemetry collection for migration tracking.

