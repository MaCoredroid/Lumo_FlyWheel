## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines the current interface as `python src/release_preview/cli.py generate` with current names `--config` and `RELEASE_PREVIEW_CONFIG`.
- `scripts/release_preview_helper.py` defines `build-preview` as a compatibility alias and forwards to the current command string `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- The previous runbook prose treated the helper path as equivalent to the current CLI and instructed operators to export `PREVIEW_SETTINGS_PATH`. That prose was overruled because live help for the current CLI exposes only `--config`, and the helper implementation marks `build-preview` as a deprecated compatibility alias.
- `tests/test_release_preview_cli.py` confirms that `--config` appears in current CLI help and `--settings` does not, while the helper alias still exists for backward compatibility.

## Remaining caveats

- Environment-variable support is not surfaced by `--help`; the current/deprecated env var names were determined from code.
- `pytest -q tests/test_release_preview_cli.py` passed, but pytest emitted a cache-write warning for `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/.pytest_cache/...` because that location is not writable in this sandbox.
