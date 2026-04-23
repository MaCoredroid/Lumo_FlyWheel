# Daily Release Preview Runbook

Use the primary CLI entrypoint:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`
2. If you prefer an environment variable, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` and run `python src/release_preview/cli.py generate`
3. For a non-mutating check, run `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`

Deprecated compatibility path:

- `python scripts/release_preview_helper.py build-preview`
- `--settings`
- `PREVIEW_SETTINGS_PATH`

Do not treat the helper alias or deprecated names as the default operator path.
