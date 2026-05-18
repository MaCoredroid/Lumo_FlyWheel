# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag with full parser-to-runtime reachability. No cleanup needed.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias flag with no standalone runtime branch. Can be deprecated once legacy deploy manifests are migrated to ENABLE_SHADOW_PREVIEW.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: operator reporting tools may still read force_legacy_seen
- rationale: Dead at runtime — parser sets the flag but no live service branches on it. Only legacy reporting labels consume it. Safe to remove from parser after confirming reporting tools no longer depend on force_legacy_seen.

