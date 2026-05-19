# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Sole live flag. Removing it would break the preview router. Must be kept as the canonical shadow preview toggle.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: docs/preview_rollout_runbook.md
- rationale: Alias with no independent runtime branch. Deprecate and redirect operators to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: tests/test_force_legacy_reporting_only.py
- rationale: Dead flag with no runtime effect. Can be removed after confirming no external deploy manifests still reference it.

