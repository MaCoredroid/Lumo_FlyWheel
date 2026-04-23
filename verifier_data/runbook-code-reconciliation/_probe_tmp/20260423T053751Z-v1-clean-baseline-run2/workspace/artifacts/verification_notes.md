## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the public parser as `release-preview` with the `generate` subcommand and exposes `--config`; `--settings` is still accepted in code but intentionally suppressed from `generate --help`, so it is deprecated rather than current.
- `src/release_preview/cli.py` resolves config in this order: `--config`, `RELEASE_PREVIEW_CONFIG`, deprecated `--settings`, deprecated `PREVIEW_SETTINGS_PATH`, then `configs/release_preview.toml`.
- `scripts/release_preview_helper.py` labels `build-preview` as a "Deprecated compatibility alias" and prints `deprecated_alias=true`, so the helper path is not the primary operator entrypoint.
- `README_fragments/legacy_path.md` was overruled by code and live help where it presented `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH` as the operator path.
- `README_fragments/current_path.md` matched the code and live help: the current path is the direct CLI with `--config` and `RELEASE_PREVIEW_CONFIG`.

## Remaining caveats

- Deprecated names still work for compatibility, so stale notes may appear functional even though they are no longer the preferred path.
- The direct CLI help does not mention default config fallback; that behavior was confirmed from code and from the `--dry-run` output.
