# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Keep the live rollout control because the runtime still branches on the shadow preview path.

## ENABLE_PREVIEW_V2

- action: `remove_after_migration`
- blockers: rollback incident requires explicit migration proof
- rationale: Do not remove the alias until manifest migration is verified post-incident; removal is only safe after migration evidence exists.

## PREVIEW_FORCE_LEGACY

- action: `do_not_remove_now`
- blockers: incident follow-up still open
- rationale: Keep the stale references documented until the rollback follow-up is complete; avoid another high-noise cleanup while the incident thread is open.

