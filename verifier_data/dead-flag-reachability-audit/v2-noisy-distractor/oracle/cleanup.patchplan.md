# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Keep the live rollout control because the runtime still branches on the shadow preview path.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: pending deploy-manifest migration
- rationale: Keep parser compatibility for now, mark it deprecated, and migrate callers to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: Remove stale docs and helper references, but do not claim live runtime cleanup because the service no longer branches on this flag.

