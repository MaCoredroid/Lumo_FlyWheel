# Daily Release Preview Runbook

Use the current CLI entrypoint:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`

If you need to supply the config path via environment instead of a flag, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`.

Deprecated compatibility path:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Deprecated names: `--settings` and `PREVIEW_SETTINGS_PATH`

The helper alias remains available for backwards compatibility, but the primary operator path is `python src/release_preview/cli.py generate`.
