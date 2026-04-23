# Daily Release Preview Runbook

Primary operator path:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Primary environment override:

- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

Observed current CLI help:

- `python src/release_preview/cli.py --help` exposes the `generate` subcommand.
- `python src/release_preview/cli.py generate --help` exposes `--config` and `--dry-run`.
- The primary `generate --help` output does not expose `--settings`.

Compatibility note:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` still exists for backward compatibility only.
- `PREVIEW_SETTINGS_PATH` and `--settings` remain deprecated compatibility inputs and must not be used as the primary runbook path.

Expected dry-run check:

- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
- Expected output includes:
  - `entrypoint=python src/release_preview/cli.py generate`
  - `config=configs/release_preview.toml`
  - `mode=dry-run`
