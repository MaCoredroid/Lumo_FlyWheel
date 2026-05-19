# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the canonical live flag with active runtime branching. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias that normalizes to ENABLE_SHADOW_PREVIEW. Deprecate after confirming all deploy manifests use ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: operator notes and reporting pipelines may still reference force_legacy_seen
- rationale: Dead flag with no runtime branch. Remove parser branch, dataclass field, legacy.py helper, and test after confirming reporting pipelines no longer depend on force_legacy_seen.

