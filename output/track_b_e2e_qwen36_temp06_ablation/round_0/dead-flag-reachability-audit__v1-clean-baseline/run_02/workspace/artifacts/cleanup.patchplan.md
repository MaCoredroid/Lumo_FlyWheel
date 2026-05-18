# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Active flag with live runtime branch. Removing it would break the shadow preview path. Keep as the canonical flag.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference it
- rationale: Live alias with no independent runtime branch. Deprecate and migrate deploy manifests to ENABLE_SHADOW_PREVIEW, then remove the parser elif branch.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: reporting and operator notes may reference the label
- rationale: Dead at runtime — parser present but no runtime branch. Remove after migrating any reporting pipelines that depend on force_legacy_seen or legacy_force_label.

