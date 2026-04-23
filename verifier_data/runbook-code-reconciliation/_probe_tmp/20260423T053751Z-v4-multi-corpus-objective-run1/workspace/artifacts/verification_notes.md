## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the current surface as `generate` with `--config` and `RELEASE_PREVIEW_CONFIG`.
- `src/release_preview/cli.py` still resolves deprecated compatibility inputs after the current ones: hidden `--settings` and `PREVIEW_SETTINGS_PATH`.
- `scripts/release_preview_helper.py` labels `build-preview` as a deprecated compatibility alias and prints a forward target of `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- The prior runbook prose that treated the helper path as a co-equal transition path and told operators to export `PREVIEW_SETTINGS_PATH` was overruled by code and live help, which present the main CLI plus `--config` as the current path.

## Remaining caveats

- The deprecated helper alias still exists and still documents `--settings`, so operators may continue to encounter old examples.
- The deprecated env var and hidden deprecated flag still work through code-level fallback, even though current CLI help does not advertise them.
