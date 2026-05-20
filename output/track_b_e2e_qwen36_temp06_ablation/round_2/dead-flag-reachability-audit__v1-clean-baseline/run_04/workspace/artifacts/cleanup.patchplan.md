# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag controlling the shadow preview runtime path. It must remain until the shadow path is permanently retired.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: This flag is an alias with no standalone behavior. It should be deprecated in favor of ENABLE_SHADOW_PREVIEW once legacy deploy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: Reporting pipeline depends on force_legacy_seen for operator labels
- rationale: This flag is parsed but has no runtime branch. Before removal, telemetry should confirm no operator tooling depends on the reporting-only label.

