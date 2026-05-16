# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Live flag with active runtime branching; core feature control

## ENABLE_PREVIEW_V2

- action: `docs_cleanup`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility; document as deprecated alias

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes reference
- rationale: Dead flag used only for telemetry; collect usage data before removal

