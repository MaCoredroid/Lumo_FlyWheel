Deploy using the current CLI path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

If configuration is supplied by environment, use `RELEASE_PREVIEW_CONFIG`.

The helper alias `python scripts/release_preview_helper.py build-preview` and the old names `--settings` / `PREVIEW_SETTINGS_PATH` remain as deprecated compatibility paths and should not be treated as the primary operator workflow.
