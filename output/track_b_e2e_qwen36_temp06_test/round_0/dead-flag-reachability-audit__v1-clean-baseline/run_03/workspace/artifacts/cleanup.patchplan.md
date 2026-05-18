# Cleanup Patch Plan

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: Verify no external deploy manifests still set PREVIEW_FORCE_LEGACY and expect the legacy-forced label, Coordinate removal of config/defaults.toml preview_force_legacy entry
- rationale: The flag is dead — parser present but no runtime branch. Remove the parser branch in load_preview_env and the force_legacy_seen field from PreviewConfig. The legacy_force_label function becomes unused and can also be removed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests still reference ENABLE_PREVIEW_V2, No migration timeline established for operators to switch to ENABLE_SHADOW_PREVIEW
- rationale: The alias is live and needed for backward compatibility with legacy deploy manifests. Add a deprecation warning in the parser branch so operators know to migrate to ENABLE_SHADOW_PREVIEW. Maintain compatibility while signaling future removal.

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the sole live standalone flag driving the shadow preview runtime branch. No cleanup needed.

