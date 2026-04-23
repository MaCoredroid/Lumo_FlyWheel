# Deploy Note

Primary release-preview invocation:

- `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Preferred environment override:

- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

Deprecated compatibility names still accepted by code:

- Helper alias: `python scripts/release_preview_helper.py build-preview`
- Deprecated flag: `--settings`
- Deprecated environment variable: `PREVIEW_SETTINGS_PATH`

Operator guidance should treat the helper path as compatibility-only and use the direct CLI entrypoint for current release-preview runs.
