Release preview should be invoked through `python src/release_preview/cli.py generate` and configured with `--config` or `RELEASE_PREVIEW_CONFIG`. The helper entrypoint `python scripts/release_preview_helper.py build-preview` remains available only as a deprecated compatibility alias for older operator habits.

Do not document the helper as an equal default path. Bundle-local code and live help show that `--settings` and `PREVIEW_SETTINGS_PATH` are deprecated fallbacks, while `configs/release_preview.toml` remains the default when no override is supplied.
