# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag driving shadow preview mode. No reason to remove.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference it
- rationale: Alias of ENABLE_SHADOW_PREVIEW. Keep parser hit for backward compat but plan removal after migration.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator dashboards may display the force_legacy_seen label
- rationale: Dead flag with no runtime branch. Verify telemetry consumers no longer depend on force_legacy_seen before removing parser code.

