# Daily Release Preview Runbook

Use either of the following operator paths during the transition:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`
2. `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Export `PREVIEW_SETTINGS_PATH=configs/release_preview.toml` before running the helper path.
