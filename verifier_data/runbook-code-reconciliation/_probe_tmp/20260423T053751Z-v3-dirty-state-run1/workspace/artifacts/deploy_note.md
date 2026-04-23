Deploy using `python src/release_preview/cli.py generate` as the primary entrypoint. The current operator-facing inputs are `--config` and `RELEASE_PREVIEW_CONFIG`.

`python scripts/release_preview_helper.py build-preview` remains available only as a deprecated compatibility alias. Its `--settings` flag and the `PREVIEW_SETTINGS_PATH` environment variable are legacy fallback names, not the preferred deployment path.
