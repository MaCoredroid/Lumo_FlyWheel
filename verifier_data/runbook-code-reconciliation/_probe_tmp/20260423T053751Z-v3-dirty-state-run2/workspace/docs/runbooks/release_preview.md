# Daily Release Preview Runbook

Use the current CLI entrypoint:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Current names:
- Entry path: `python src/release_preview/cli.py generate`
- Flag: `--config`
- Environment variable: `RELEASE_PREVIEW_CONFIG`

Deprecated compatibility names still exist but are not the primary operator path:
- Alias: `python scripts/release_preview_helper.py build-preview`
- Deprecated flag: `--settings`
- Deprecated environment variable: `PREVIEW_SETTINGS_PATH`

If no flag or environment variable is supplied, the CLI resolves the config path to `configs/release_preview.toml`.
