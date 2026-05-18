# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag controlling the preview router. It must remain active and functional.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference it
- rationale: Alias with no independent runtime branch. Can be deprecated once legacy deploy manifests are migrated to use ENABLE_SHADOW_PREVIEW directly.

## PREVIEW_FORCE_LEGACY

- action: `telemetry_first`
- blockers: operator notes and reporting pipelines may depend on the force_legacy_seen signal
- rationale: Parser hit exists but no runtime branch. Before removal, verify that reporting pipelines and operator tooling no longer depend on the force_legacy_seen flag in the plan output.

