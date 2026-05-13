# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Primary live flag controlling preview rollout. Must be retained for operational control.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference this flag
- rationale: Alias flag with no standalone value. Deprecate after confirming no active deployments depend on it.

## PREVIEW_FORCE_LEGACY

- action: `docs_cleanup`
- blockers: need to update operator runbooks and remove references to this flag
- rationale: Flag is dead for runtime branching but may be referenced in operator docs and runbooks. Clean up documentation and telemetry labels.

