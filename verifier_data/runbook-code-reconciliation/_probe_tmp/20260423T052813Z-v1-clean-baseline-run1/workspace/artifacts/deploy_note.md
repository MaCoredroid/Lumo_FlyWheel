Use `python src/release_preview/cli.py generate --config configs/release_preview.toml` for the daily release-preview run. `RELEASE_PREVIEW_CONFIG` is the current environment fallback when a flag is not provided.

The helper path `python scripts/release_preview_helper.py build-preview` and the legacy names `--settings` / `PREVIEW_SETTINGS_PATH` remain available only for compatibility and should not be used as the primary operator path.
