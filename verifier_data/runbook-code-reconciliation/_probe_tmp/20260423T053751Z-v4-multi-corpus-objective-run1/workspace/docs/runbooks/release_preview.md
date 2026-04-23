# Daily Release Preview Runbook

Use the current release-preview CLI as the primary operator path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

You can also provide the config path through the current environment variable:

`RELEASE_PREVIEW_CONFIG=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run`

Legacy compatibility remains available through the helper alias, but it is not the preferred runbook path:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Notes:

1. Live help for `python src/release_preview/cli.py generate --help` exposes `--config` and does not expose `--settings`.
2. Code in `src/release_preview/cli.py` still accepts deprecated compatibility names in resolution order: `--settings` and `PREVIEW_SETTINGS_PATH`.
3. The helper script advertises `build-preview` as a deprecated compatibility alias and forwards operators to the current CLI command.
