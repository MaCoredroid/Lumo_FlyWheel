# Daily Release Preview Runbook

Use the bundle-local Python CLI as the primary operator path.

Current operator step:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Current environment variable override:

- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

Compatibility-only legacy path:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Notes:

- `python src/release_preview/cli.py --help` exposes `generate` as the active subcommand.
- `python src/release_preview/cli.py generate --help` exposes `--config` and does not expose `--settings`.
- `PREVIEW_SETTINGS_PATH` is a deprecated fallback env var retained for compatibility, not the primary operator instruction.
