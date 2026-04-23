# Preview Rollout Runbook

The live preview router branches on `preview_runtime_branch:shadow_preview_path`
when `ENABLE_SHADOW_PREVIEW` is enabled.

`ENABLE_PREVIEW_V2` is still accepted by the env parser for legacy deploy
manifests, but it normalizes to the same shadow path and is not a standalone
runtime branch.

`PREVIEW_FORCE_LEGACY` is left in reporting and operator notes only. The live
service does not branch on it anymore.
