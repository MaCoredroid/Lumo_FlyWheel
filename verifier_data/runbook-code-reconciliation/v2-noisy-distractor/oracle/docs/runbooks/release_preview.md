# Daily Release Preview Runbook

Use the current primary CLI path for the daily release preview:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`

If you prefer the env-backed form, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` and run `python src/release_preview/cli.py generate`.

## Deprecated compatibility path

The helper alias `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` still exists for backwards compatibility, but it is compatibility-only and must not be used as the primary runbook step.
