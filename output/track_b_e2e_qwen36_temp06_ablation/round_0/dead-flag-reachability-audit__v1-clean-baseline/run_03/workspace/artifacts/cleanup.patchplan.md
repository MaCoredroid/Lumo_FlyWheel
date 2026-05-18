# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag — parser and runtime branch both active and tested.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: Legacy deploy manifests may still set ENABLE_PREVIEW_V2
- rationale: Alias flag normalizing to ENABLE_SHADOW_PREVIEW; safe to remove after legacy deploy manifests are updated.

## PREVIEW_FORCE_LEGACY

- action: `remove_after_migration`
- blockers: Confirm no external tooling depends on force_legacy_seen reporting
- rationale: Dead flag — parser sets a reporting-only field that no runtime branch reads. Safe to remove after confirming no external tooling reads force_legacy_seen.

