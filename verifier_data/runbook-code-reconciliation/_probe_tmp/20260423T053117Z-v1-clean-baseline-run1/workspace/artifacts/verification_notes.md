## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the primary entrypoint as `python src/release_preview/cli.py generate`; the helper script only prints a compatibility forward target and marks itself with `deprecated_alias=true`.
- `src/release_preview/cli.py` defines `--config` as current and `--settings` as deprecated. Live help confirms `--config` is shown on the current path and `--settings` is hidden there.
- `src/release_preview/cli.py` resolves `RELEASE_PREVIEW_CONFIG` before the deprecated `PREVIEW_SETTINGS_PATH`, so the current env var is `RELEASE_PREVIEW_CONFIG` and the old env var is deprecated compatibility only.
- The prior runbook prose and the legacy README fragment pointed operators to `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH`; that prose was overruled by code and live help.

## Remaining caveats

- Environment-variable precedence was determined from code inspection, not from separate live env-var mutation runs.
- `pytest -q tests/test_release_preview_cli.py` passed, but pytest emitted a cache-write warning for a parent-level `.pytest_cache` path outside the writable sandbox.
