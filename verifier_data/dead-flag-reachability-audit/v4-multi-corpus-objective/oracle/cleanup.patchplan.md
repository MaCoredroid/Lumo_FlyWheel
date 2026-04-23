# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Keep the live rollout control because the runtime still branches on the shadow preview path.

## ENABLE_PREVIEW_V2

- action: `telemetry_first`
- blockers: release freeze active, pending deploy-manifest migration
- rationale: During the freeze, keep parser compatibility and add telemetry or docs before any removal work.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: Remove stale docs and helper references, but do not claim live runtime cleanup because the service no longer branches on this flag.

