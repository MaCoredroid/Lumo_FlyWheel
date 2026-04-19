# Release Sync Marketplace Alignment Report

- Updated `.codex-plugin/plugin.json` to schema version 2.
- Standardized the plugin slug to `release-sync`.
- Standardized the startup command to `python -m release_sync_mcp`.
- Corrected the manifest icon path to `assets/icon-release-sync.png`.
- Removed the retired tool alias so only the canonical marketplace tool id remains.

Remaining risk: the visible workspace does not include the runtime package, packaged assets, or real validator and test harness, so bundle build integrity and the module smoke path still need confirmation in the full evaluator workspace.
