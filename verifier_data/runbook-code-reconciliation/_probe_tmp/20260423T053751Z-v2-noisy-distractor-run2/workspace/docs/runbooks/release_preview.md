# Daily Release Preview Runbook

Primary operator path:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`
2. If you need env-based config selection, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` and run `python src/release_preview/cli.py generate`
3. Add `--dry-run` when you want to verify the resolved config without performing a full preview build

Compatibility notes:

- `python scripts/release_preview_helper.py build-preview` is a deprecated compatibility alias, not the primary operator path
- `--config` is the current flag; `--settings` is deprecated
- `RELEASE_PREVIEW_CONFIG` is the current env var; `PREVIEW_SETTINGS_PATH` is deprecated
