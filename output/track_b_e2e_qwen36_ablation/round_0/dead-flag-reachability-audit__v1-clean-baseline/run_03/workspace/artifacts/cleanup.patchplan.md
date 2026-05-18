# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Active standalone flag that controls the shadow preview runtime path; removing it would break the preview rollout control plane.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias that normalizes to ENABLE_SHADOW_PREVIEW with no unique runtime effect. Safe to remove from parser once all legacy deploy manifests are migrated to use ENABLE_SHADOW_PREVIEW directly.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: Operator dashboards may surface the force_legacy_seen field for reporting
- rationale: Dead flag with no runtime branching — only sets a reporting label in legacy.py. Parser hit in config.py can be removed after confirming no operator tooling depends on the force_legacy_seen telemetry field.

