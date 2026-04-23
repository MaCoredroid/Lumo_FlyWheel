# Daily Release Preview Runbook

Use the current CLI as the primary operator path:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
2. If you prefer an environment variable, set `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` and omit `--config`.
3. Add `--dry-run` only when you want the preview generated in dry-run mode.

Configuration resolution in the current CLI is:

1. `--config`
2. `RELEASE_PREVIEW_CONFIG`
3. deprecated `--settings`
4. deprecated `PREVIEW_SETTINGS_PATH`
5. default `configs/release_preview.toml`

Compatibility note:

- `python scripts/release_preview_helper.py build-preview` is a deprecated compatibility alias, not the preferred runbook path.
- `--settings` and `PREVIEW_SETTINGS_PATH` remain accepted for compatibility, but operators should use `--config` and `RELEASE_PREVIEW_CONFIG`.
