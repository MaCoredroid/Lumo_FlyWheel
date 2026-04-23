# Daily Release Preview Runbook

Use the current CLI entrypoint, not the compatibility helper alias.

## Primary operator path

Run:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Optional current environment fallback:

`RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

## Compatibility notes

- The helper alias `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` still exists for backwards compatibility.
- `--settings` and `PREVIEW_SETTINGS_PATH` are deprecated names. They are not the primary runbook path.
- Live help for `python src/release_preview/cli.py generate --help` exposes `--config` and does not expose `--settings`.
