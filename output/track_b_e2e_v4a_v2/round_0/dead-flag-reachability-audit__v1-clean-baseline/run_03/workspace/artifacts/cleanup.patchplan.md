# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with full runtime reachability. Must be retained for active preview rollout control.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag for backward compatibility. Should be deprecated with migration path to ENABLE_SHADOW_PREVIEW once legacy manifests are updated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator notes reference
- rationale: Dead flag with no runtime effect. Remove from env parser and update documentation to clarify it only affects reporting labels.

