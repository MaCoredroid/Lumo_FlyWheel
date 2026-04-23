# Daily Release Preview Runbook

Use the direct CLI as the primary operator path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Accepted configuration sources resolve in this order:

1. `--config`
2. `RELEASE_PREVIEW_CONFIG`
3. deprecated `--settings`
4. deprecated `PREVIEW_SETTINGS_PATH`
5. default `configs/release_preview.toml`

The compatibility helper remains available, but it is not the primary path:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Treat the helper as a deprecated alias that forwards operators back to the direct CLI flow.
