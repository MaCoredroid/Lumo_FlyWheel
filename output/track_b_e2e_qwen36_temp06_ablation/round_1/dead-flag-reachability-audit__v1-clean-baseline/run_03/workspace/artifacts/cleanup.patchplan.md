# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling the preview shadow path. No cleanup needed at this time.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias flag with no standalone runtime branch. Deprecate in favor of ENABLE_SHADOW_PREVIEW once legacy deploy manifests are updated.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: Operator reporting pipelines may depend on force_legacy_seen telemetry
- rationale: Dead for runtime branching but still parsed for reporting. Collect telemetry on force_legacy_seen usage before removal.

