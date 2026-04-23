Deploy using the current CLI entrypoint: `python src/release_preview/cli.py generate`.

Use `--config` or `RELEASE_PREVIEW_CONFIG` for the config path. If neither is set, the CLI falls back to `configs/release_preview.toml`.

The helper path `python scripts/release_preview_helper.py build-preview` is deprecated compatibility-only behavior. Its `--settings` flag and the `PREVIEW_SETTINGS_PATH` environment variable are legacy names and should not be used as the default release-preview runbook path.
