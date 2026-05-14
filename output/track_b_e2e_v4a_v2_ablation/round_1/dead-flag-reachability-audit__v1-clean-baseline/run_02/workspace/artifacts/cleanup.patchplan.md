# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with full runtime reachability. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `docs_cleanup`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility. Document that it normalizes to ENABLE_SHADOW_PREVIEW. Remove when legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator reporting workflows
- rationale: Dead flag kept for telemetry. Monitor usage before removal. Update operator notes to clarify it has no runtime effect.

