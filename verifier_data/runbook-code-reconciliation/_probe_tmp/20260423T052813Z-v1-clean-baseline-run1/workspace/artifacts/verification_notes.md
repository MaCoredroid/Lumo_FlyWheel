## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines `generate` as the active subcommand under `release-preview`, with `--config` as the current flag and `RELEASE_PREVIEW_CONFIG` as the current environment fallback.
- `src/release_preview/cli.py` still accepts deprecated `--settings` and `PREVIEW_SETTINGS_PATH`, but `--settings` is hidden from live `generate --help`, which matches the tests and indicates deprecation rather than current operator guidance.
- `scripts/release_preview_helper.py` labels `build-preview` as a deprecated compatibility alias and forwards operators to `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- Conflicting README fragment prose that still presents `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH` as the main path was overruled by bundle-local code and live help.

## Remaining caveats

- The current and deprecated environment variable names are not surfaced in CLI help; they were determined from `src/release_preview/cli.py`.
- The helper alias still has visible `--settings` help for backwards compatibility, so operators can still discover the legacy path even though it is no longer primary.
