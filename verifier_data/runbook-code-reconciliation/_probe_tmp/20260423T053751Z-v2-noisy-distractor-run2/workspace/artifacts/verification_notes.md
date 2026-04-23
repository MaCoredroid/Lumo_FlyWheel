## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` makes `generate` the active command path and resolves config in this order: `--config`, `RELEASE_PREVIEW_CONFIG`, deprecated `--settings`, deprecated `PREVIEW_SETTINGS_PATH`, then `configs/release_preview.toml`
- `src/release_preview/cli.py generate --help` shows `--config` and suppresses `--settings`, so the hidden legacy flag remains backward-compatible but is not the current operator-facing interface
- Conflicting README prose was overruled by code and live help: the legacy fragment still instructs operators to use `python scripts/release_preview_helper.py build-preview --settings ...` and `PREVIEW_SETTINGS_PATH`, but the helper code marks `build-preview` as a deprecated compatibility alias and the primary CLI help exposes `generate` with `--config`
- The current README fragment also notes that an older example kept `PREVIEW_SETTINGS_PATH`; code confirms the current env var is `RELEASE_PREVIEW_CONFIG`

## Remaining caveats

- The deprecated helper alias and deprecated names still function, so operator copy-paste from older notes can continue to work even though it is no longer the preferred path
- The current env var is implemented in code but does not appear in CLI help text, so operators must rely on code-backed runbook guidance for that detail
