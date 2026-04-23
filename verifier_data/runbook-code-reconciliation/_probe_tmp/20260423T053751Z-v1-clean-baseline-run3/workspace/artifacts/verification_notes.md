## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines `generate` as the only subcommand on the primary `release-preview` CLI and prints `entrypoint=python src/release_preview/cli.py generate`, so `python src/release_preview/cli.py generate` is the current primary path.
- `src/release_preview/cli.py` treats `--config` and `RELEASE_PREVIEW_CONFIG` as current names, while `--settings` is hidden from `generate --help` and only accepted as a deprecated fallback.
- `src/release_preview/cli.py` resolves `PREVIEW_SETTINGS_PATH` only after `RELEASE_PREVIEW_CONFIG`, which marks `PREVIEW_SETTINGS_PATH` as deprecated compatibility rather than the preferred env var.
- `scripts/release_preview_helper.py` labels `build-preview` as `Deprecated compatibility alias` and forwards to `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- The prior README-style runbook prose that instructed operators to use `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH` was overruled by bundle-local code and live help.

## Remaining caveats

- The helper alias still appears in its own help output for compatibility, so older automation may continue to function even though it is not the primary documented path.
- The primary CLI help does not surface the deprecated `--settings` fallback, so operators should not rely on it for new usage.
