## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the active flow as the `generate` subcommand on the `release-preview` CLI and exposes `--config` in live help, while the deprecated `--settings` argument is hidden with `argparse.SUPPRESS`.
- `src/release_preview/cli.py` resolves configuration in this order: `--config`, `RELEASE_PREVIEW_CONFIG`, deprecated `--settings`, deprecated `PREVIEW_SETTINGS_PATH`, then `configs/release_preview.toml`.
- `scripts/release_preview_helper.py` defines `build-preview` as a deprecated compatibility alias and prints `deprecated_alias=true` while forwarding operators toward `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- Conflicting README fragment prose that presented both operator paths as equivalent was overruled by bundle-local code and live help: only the main CLI exposes the current flag, while the helper advertises a deprecated compatibility alias.

## Remaining caveats

- The deprecated helper alias is still present and still accepts `--settings`, so older automation may continue to work even though it should not be documented as the primary path.
- The deprecated environment variable `PREVIEW_SETTINGS_PATH` is still accepted in code, but it is not surfaced by CLI help and should be treated as a fallback only.
