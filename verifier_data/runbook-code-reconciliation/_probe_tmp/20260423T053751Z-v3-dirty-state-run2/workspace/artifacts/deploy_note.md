Deploy using the current CLI path: `python src/release_preview/cli.py generate --config configs/release_preview.toml`.

Treat `python scripts/release_preview_helper.py build-preview`, `--settings`, and `PREVIEW_SETTINGS_PATH` as deprecated compatibility surfaces only. The live CLI help exposes `--config`, and the code resolves `RELEASE_PREVIEW_CONFIG` ahead of deprecated names.
