# Daily Release Preview Runbook

Use the bundle-local CLI directly for the daily release preview.

## Primary path

Run the current entrypoint:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

If you prefer an environment variable instead of the flag, set:

`RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

If neither `--config` nor `RELEASE_PREVIEW_CONFIG` is provided, the CLI falls back to:

`configs/release_preview.toml`

## Verification shortcut

To confirm the resolved config without performing a full run, use:

`python src/release_preview/cli.py generate --dry-run`

## Deprecated compatibility path

The helper alias remains available for backwards compatibility only:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Treat `build-preview`, `--settings`, and `PREVIEW_SETTINGS_PATH` as deprecated names rather than the primary operator workflow.
