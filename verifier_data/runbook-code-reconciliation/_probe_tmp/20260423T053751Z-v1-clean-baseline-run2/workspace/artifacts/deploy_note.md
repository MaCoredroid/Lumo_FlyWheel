Use `python src/release_preview/cli.py generate` as the release-preview entrypoint. Pass config with `--config` or `RELEASE_PREVIEW_CONFIG`; if neither is set, the CLI falls back to `configs/release_preview.toml`.

Do not treat `python scripts/release_preview_helper.py build-preview`, `--settings`, or `PREVIEW_SETTINGS_PATH` as the primary path. They remain available only as deprecated compatibility names.
