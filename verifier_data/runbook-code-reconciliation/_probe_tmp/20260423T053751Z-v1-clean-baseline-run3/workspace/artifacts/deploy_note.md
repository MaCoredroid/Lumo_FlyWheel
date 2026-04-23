Use `python src/release_preview/cli.py generate` as the release-preview entrypoint going forward.

Pass config with `--config configs/release_preview.toml` or set `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`.

The helper path `python scripts/release_preview_helper.py build-preview --settings ...` remains available only as a deprecated compatibility alias. Treat `--settings` and `PREVIEW_SETTINGS_PATH` as deprecated names when updating operator docs or automation.
