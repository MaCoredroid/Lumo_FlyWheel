# Daily Release Preview Runbook

Use the current release-preview CLI as the primary operator path:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`

You can also set `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` instead of passing `--config`.

Deprecated compatibility surfaces still exist but are not the primary runbook path:

- Legacy helper alias: `python scripts/release_preview_helper.py build-preview`
- Deprecated flag: `--settings`
- Deprecated environment variable: `PREVIEW_SETTINGS_PATH`
