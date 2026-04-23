# Daily Release Preview Runbook

Use the current CLI directly:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Supported current config paths:

- Preferred flag: `--config`
- Preferred env fallback: `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`
- Built-in default when neither is set: `configs/release_preview.toml`

Optional dry-run check:

- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`

Do not use the legacy helper as the default operator path:

- `python scripts/release_preview_helper.py build-preview` is a deprecated compatibility alias that forwards to the current CLI.
- `--settings` is deprecated in favor of `--config`.
- `PREVIEW_SETTINGS_PATH` is deprecated in favor of `RELEASE_PREVIEW_CONFIG`.
