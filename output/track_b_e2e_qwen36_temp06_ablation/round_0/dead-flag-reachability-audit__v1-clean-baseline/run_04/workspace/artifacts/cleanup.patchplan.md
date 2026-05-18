# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag controlling the shadow preview path. Removing it would break the active rollout mechanism.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: This flag is a live alias to ENABLE_SHADOW_PREVIEW. It should be deprecated with a migration path for legacy deploy manifests that still reference it.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator reporting depends on force_legacy_seen label
- rationale: This flag is dead at runtime but still parsed for reporting. Before removal, telemetry should confirm no operators depend on the force_legacy_seen reporting label.

