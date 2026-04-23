# Daily Release Preview Runbook

Use the current CLI as the primary operator path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Current names:

- Flag: `--config`
- Environment variable: `RELEASE_PREVIEW_CONFIG`
- Default config path when no flag or environment override is supplied: `configs/release_preview.toml`

Deprecated compatibility surface:

- Helper alias: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Deprecated flag name: `--settings`
- Deprecated environment variable fallback in the main CLI: `PREVIEW_SETTINGS_PATH`

Do not treat the helper alias as an equal operator path. Bundle-local help and code mark it as compatibility-only; use it only for older callers that have not been migrated to `python src/release_preview/cli.py generate`.
