# Daily Release Preview Runbook

Current operator step:

1. Run `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
2. If you prefer an environment variable instead of an explicit flag, export `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml` and then run `python src/release_preview/cli.py generate`.

Compatibility-only legacy path:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Deprecated names retained for compatibility:

- Flag: `--settings`
- Environment variable: `PREVIEW_SETTINGS_PATH`

Why this runbook uses the `src/release_preview/cli.py` path:

- Live help for `python src/release_preview/cli.py --help` exposes the `generate` subcommand.
- Live help for `python src/release_preview/cli.py generate --help` exposes `--config` and does not expose `--settings`.
- Bundle-local code in `src/release_preview/cli.py` resolves `--config` / `RELEASE_PREVIEW_CONFIG` first, while `scripts/release_preview_helper.py` labels `build-preview` as a deprecated compatibility alias.
