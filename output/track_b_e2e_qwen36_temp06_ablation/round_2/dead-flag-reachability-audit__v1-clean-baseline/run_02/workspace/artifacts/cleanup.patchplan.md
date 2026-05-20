# Cleanup Patch Plan

## ENABLE_SHADOW_PREVIEW

- action: `keep`
- blockers: none
- rationale: Live and actively used for shadow preview routing.

## ENABLE_PREVIEW_V2

- action: `deprecate`
- blockers: legacy deploy manifests may still reference ENABLE_PREVIEW_V2
- rationale: Alias normalizing to ENABLE_SHADOW_PREVIEW; deprecate in favor of canonical flag.

## PREVIEW_FORCE_LEGACY

- action: `do_not_remove_now`
- blockers: operator tooling may depend on the legacy-forced reporting label, no migration path defined for force_legacy_seen consumers
- rationale: Parser present but no runtime branch; safe to keep for now as reporting-only.

