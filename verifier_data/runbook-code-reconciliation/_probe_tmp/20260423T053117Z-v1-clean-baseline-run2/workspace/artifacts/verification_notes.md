## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the primary operator path as `python src/release_preview/cli.py generate`, with `--config` and `RELEASE_PREVIEW_CONFIG` as the current names.
- `src/release_preview/cli.py` still accepts `--settings` and `PREVIEW_SETTINGS_PATH`, but only as deprecated compatibility fallbacks behind the current flag and env var in config resolution order.
- `scripts/release_preview_helper.py` marks `build-preview` as a deprecated compatibility alias and prints a forward target to `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- The prior prose in `docs/runbooks/release_preview.md` and `README_fragments/legacy_path.md` was overruled by code and live help because it treated the helper plus `--settings` and `PREVIEW_SETTINGS_PATH` as the default operator path, while the bundle-local CLI exposes `generate` with `--config` as the current interface.

## Remaining caveats

- The helper alias still works for backwards compatibility, so operators may still encounter `build-preview` in older notes even though it is not the primary path.
- The reconciliation here verifies command surface, precedence, and compatibility behavior; it does not validate the contents of `configs/release_preview.toml` beyond the dry-run output.
