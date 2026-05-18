# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the active live flag controlling the preview router branch. Removing it would break the shadow preview path entirely.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests still reference ENABLE_PREVIEW_V2
- rationale: Live alias retained for legacy deploy manifests only. Should be deprecated with a migration path to ENABLE_SHADOW_PREVIEW before removal.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: Operator notes still reference PREVIEW_FORCE_LEGACY for historical context
- rationale: Parser presence with no runtime branching. Only used for reporting labels. Keep telemetry while cleaning up docs to mark as dead.

