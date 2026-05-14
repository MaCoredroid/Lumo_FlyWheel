# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with full parser and runtime reachability. Essential for shadow preview functionality.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference this flag
- rationale: Alias flag kept for backward compatibility. Should be deprecated with migration path to ENABLE_SHADOW_PREVIEW once legacy manifests are updated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator notes reference this flag, telemetry reports may still include force_legacy_seen
- rationale: Dead flag with no runtime effect. Remove from parser after updating operator documentation and telemetry to avoid confusion about its purpose.

