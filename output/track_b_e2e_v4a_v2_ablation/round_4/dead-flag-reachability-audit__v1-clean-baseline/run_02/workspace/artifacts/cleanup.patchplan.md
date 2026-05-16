# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with active runtime branch. Must remain for production preview routing.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility. Can be deprecated once legacy manifests migrate to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes dependency
- rationale: Dead flag with no runtime impact. Remove parser after confirming telemetry and operator notes no longer depend on it.

