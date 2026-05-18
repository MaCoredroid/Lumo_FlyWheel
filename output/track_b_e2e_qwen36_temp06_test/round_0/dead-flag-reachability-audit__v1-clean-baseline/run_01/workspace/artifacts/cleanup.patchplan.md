# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag actively branching the runtime. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias flag that normalizes to ENABLE_SHADOW_PREVIEW. Can be deprecated once deploy manifests migrate to the canonical flag name.

## PREVIEW_FORCE_LEGACY

- action: `do_not_remove_now`
- blockers: reporting pipeline may still consume force_legacy_seen telemetry
- rationale: Dead flag with no runtime branch but still parsed for reporting. Removing the parser would break legacy_force_label reporting. Keep until reporting migration is complete.

