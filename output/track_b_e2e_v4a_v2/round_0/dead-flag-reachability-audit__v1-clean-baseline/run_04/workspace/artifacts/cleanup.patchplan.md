# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Active live flag with both parser and runtime reachability; core control for preview shadow mode.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Alias flag maintained for backward compatibility; deprecate once legacy manifests migrate completes.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes reference
- rationale: Dead flag with parser presence; collect telemetry on usage before removal to avoid breaking operator workflows.

