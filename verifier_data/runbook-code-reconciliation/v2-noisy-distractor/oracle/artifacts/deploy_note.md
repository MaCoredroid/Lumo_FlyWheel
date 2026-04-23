# Deploy Note

Updated the release-preview runbook so operators now use `python src/release_preview/cli.py generate` with `--config` / `RELEASE_PREVIEW_CONFIG` as the primary path. The legacy helper alias and `--settings` / `PREVIEW_SETTINGS_PATH` remain compatibility-only and are no longer presented as the default instructions.
