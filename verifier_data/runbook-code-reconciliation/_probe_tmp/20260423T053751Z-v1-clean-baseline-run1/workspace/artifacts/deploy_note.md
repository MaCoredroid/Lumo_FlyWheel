Use the current release-preview entrypoint:

- `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Current config environment variable:

- `RELEASE_PREVIEW_CONFIG`

Deprecated compatibility path retained, but not primary:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Deprecated flag: `--settings`
- Deprecated environment variable: `PREVIEW_SETTINGS_PATH`
