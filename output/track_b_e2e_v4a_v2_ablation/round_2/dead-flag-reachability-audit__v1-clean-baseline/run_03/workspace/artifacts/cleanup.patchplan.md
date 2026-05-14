# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag for the preview feature. It has full parser and runtime support.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: This is an alias for ENABLE_SHADOW_PREVIEW kept only for legacy deploy manifests. Should be deprecated with migration path.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes dependency
- rationale: This flag is dead for runtime but still used in reporting and operator notes. Collect telemetry before removal.

