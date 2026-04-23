# Daily Release Preview Runbook

Use the current bundle-local CLI as the default operator path:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

Notes:

- The current flag is `--config`.
- The current environment variable is `RELEASE_PREVIEW_CONFIG`.
- If neither is provided, the CLI falls back to `configs/release_preview.toml`.

Compatibility-only legacy path:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Treat the helper path as deprecated compatibility support, not the default workflow. The helper prints a forward target back to the primary CLI path, and live help keeps `--settings` on the helper while hiding `--settings` from the primary `generate` help.
