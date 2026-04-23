Use `python src/release_preview/cli.py generate --config configs/release_preview.toml` as the release-preview entrypoint for this bundle. Prefer `RELEASE_PREVIEW_CONFIG` for env-based config injection when needed.

Treat `python scripts/release_preview_helper.py build-preview`, `--settings`, and `PREVIEW_SETTINGS_PATH` as deprecated compatibility surfaces only. They still resolve through the current code path, but they should not be used as the default release-preview instructions.
