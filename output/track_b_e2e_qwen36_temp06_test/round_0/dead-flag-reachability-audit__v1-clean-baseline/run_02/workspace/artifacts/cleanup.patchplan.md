# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag controlling the shadow preview path. It must be kept as-is since it is the sole runtime branch driver.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: This is a live alias that normalizes to ENABLE_SHADOW_PREVIEW. It should be deprecated in docs and eventually removed once all deploy manifests migrate to the canonical flag name.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operators may rely on the reporting label in legacy.py
- rationale: This flag is parsed but has no runtime branch effect. Before removing it, telemetry should confirm no operators depend on the force_legacy_seen reporting label. The parser hit and legacy label function should be removed together.

