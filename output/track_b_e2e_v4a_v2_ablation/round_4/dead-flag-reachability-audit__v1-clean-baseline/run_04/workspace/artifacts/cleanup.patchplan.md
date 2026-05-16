# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: This is the primary live flag controlling the preview feature. It has both parser presence and runtime reachability.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests
- rationale: This is an alias for ENABLE_SHADOW_PREVIEW kept for backward compatibility. Should be deprecated once legacy manifests are migrated.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: none
- rationale: This flag has no runtime effect and should be removed from documentation and operator notes. The parser can remain for telemetry but should be clearly marked as deprecated.

