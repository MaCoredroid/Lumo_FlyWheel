## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
- `python src/release_preview/cli.py generate --settings configs/release_preview.toml`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines `generate` on the main CLI, prints `entrypoint=python src/release_preview/cli.py generate`, and resolves config in this order: `--config`, `RELEASE_PREVIEW_CONFIG`, `--settings`, `PREVIEW_SETTINGS_PATH`, then the default file.
- `scripts/release_preview_helper.py` labels `build-preview` as `Deprecated compatibility alias` in live help and prints `deprecated_alias=true`, so the helper remains supported but is not the primary operator path.
- `README_fragments/legacy_path.md` presents `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH` as the operator note, but that prose was overruled by code and live help showing the helper alias is deprecated.
- `README_fragments/current_path.md` correctly points to `python src/release_preview/cli.py generate`, but its carried-over `PREVIEW_SETTINGS_PATH` example was overruled by code, which makes `RELEASE_PREVIEW_CONFIG` the current environment variable and keeps `PREVIEW_SETTINGS_PATH` only as a deprecated fallback.

## Remaining caveats

- The deprecated `--settings` flag is intentionally hidden from `python src/release_preview/cli.py generate --help`, so it still works but should not appear in operator-facing instructions.
- `pytest -q tests/test_release_preview_cli.py` passed, but emitted a cache-write warning because pytest could not write to a parent `.pytest_cache` path outside this sandbox.
