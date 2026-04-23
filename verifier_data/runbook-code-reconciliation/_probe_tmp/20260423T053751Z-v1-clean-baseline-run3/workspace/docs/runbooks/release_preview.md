# Daily Release Preview Runbook

Primary operator step:

1. `python src/release_preview/cli.py generate --config configs/release_preview.toml`
2. If you prefer env-based config resolution, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` instead of passing `--config`.
3. Add `--dry-run` when you want to verify config resolution without performing the normal generate path.

Legacy compatibility path:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Treat this helper path as deprecated compatibility only. The primary operator entrypoint is `python src/release_preview/cli.py generate`.
- Treat `--settings` and `PREVIEW_SETTINGS_PATH` as deprecated names retained for older callers.
