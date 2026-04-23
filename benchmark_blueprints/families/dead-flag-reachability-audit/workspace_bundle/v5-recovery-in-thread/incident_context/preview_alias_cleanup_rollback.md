# Incident: Preview Alias Cleanup Rollback

On 2026-04-11, a cleanup removed `ENABLE_PREVIEW_V2` before all deploy manifests
were migrated. Staging silently stopped activating the shadow preview path. The
cleanup was rolled back the same day. Future audits must mention the rollback
before proposing any removal.
