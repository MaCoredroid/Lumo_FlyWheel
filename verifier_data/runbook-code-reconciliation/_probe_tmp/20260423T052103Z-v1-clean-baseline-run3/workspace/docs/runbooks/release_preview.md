# Daily Release Preview Runbook

This runbook is reconciled against the bundle-local CLI in `src/release_preview/cli.py`.

## Primary operator path

Use the current CLI entrypoint:

1. Optional: export the current config environment variable if you do not want to pass the flag inline:
   `export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`
2. Run the preview generation command:
   `python src/release_preview/cli.py generate --config configs/release_preview.toml`
3. For a non-mutating verification pass, add `--dry-run`:
   `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`

## Current interface

- Subcommand: `generate`
- Current flag: `--config`
- Current environment variable: `RELEASE_PREVIEW_CONFIG`
- Default config path when no flag or environment variable is provided: `configs/release_preview.toml`
- Optional verification mode: `--dry-run`

## Compatibility notes

The helper alias remains available only for backward compatibility:

- Helper script: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Deprecated flag on the primary CLI: `--settings`
- Deprecated environment variable on the primary CLI: `PREVIEW_SETTINGS_PATH`

Do not use the helper alias, deprecated flag, or deprecated environment variable as the primary documented path.

## CLI help checkpoints

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
