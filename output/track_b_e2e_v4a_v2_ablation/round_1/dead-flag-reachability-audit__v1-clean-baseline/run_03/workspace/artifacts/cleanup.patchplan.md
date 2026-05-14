# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with active runtime branching. Essential for preview rollout control.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: Live alias with parser support but no standalone branch. Deprecate after legacy manifests migrate.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes dependency
- rationale: Dead flag with parser presence but no runtime branch. Remove parser after confirming telemetry and notes no longer depend on it.

