# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with direct runtime branch. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Should be deprecated in docs and parser warnings added before eventual removal to give deploy manifests time to migrate.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: Reporting pipeline depends on force_legacy_seen for operator labels
- rationale: Parser present but no runtime branch. Before removal, verify that the reporting-only label from legacy_force_label is not consumed by any downstream telemetry or alerting system.

