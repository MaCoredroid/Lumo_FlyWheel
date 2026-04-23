# Daily Release Preview Runbook

Current operator step:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Current environment fallback:

- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

Compatibility note:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` still exists as a deprecated compatibility alias.
- `PREVIEW_SETTINGS_PATH` is deprecated and only retained as a fallback behind the current flag and current environment variable.
