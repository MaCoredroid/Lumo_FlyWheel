# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling the preview router. Essential for runtime branching.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Can be deprecated once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator notes may reference this flag
- rationale: Flag is dead for runtime but may be referenced in operator documentation. Clean up docs before removing parser.

