# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Canonical live flag with confirmed parser and runtime branch. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still reference this env var
- rationale: Live alias with no independent runtime branch. Should be deprecated in favor of ENABLE_SHADOW_PREVIEW once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: Operator notes and reporting still reference the flag
- rationale: Dead flag with only reporting-level usage. Collect telemetry on deploy manifests that still set this before removing from parser and defaults.

