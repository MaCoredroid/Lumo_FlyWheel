# Deploy Note

Release preview generation should be operated through the current CLI path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Preferred configuration source:

- Flag: `--config`
- Environment variable: `RELEASE_PREVIEW_CONFIG`

Deprecated compatibility surfaces remain present but should not drive the runbook:

- Helper alias: `python scripts/release_preview_helper.py build-preview`
- Deprecated flag: `--settings`
- Deprecated environment variable: `PREVIEW_SETTINGS_PATH`

The helper path is retained for backwards compatibility only and forwards users toward the current CLI command.
