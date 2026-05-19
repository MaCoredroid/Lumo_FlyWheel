# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling the preview runtime branch. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias only. Normalize all deploy manifests to ENABLE_SHADOW_PREVIEW, then remove the elif branch in load_preview_env.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator notes and reporting dashboards may still display force_legacy_seen in build_preview_plan output
- rationale: Parser branch and legacy_force_label are reporting-only. Remove from config parsing once operator dashboards are updated, and clean up legacy.py.

