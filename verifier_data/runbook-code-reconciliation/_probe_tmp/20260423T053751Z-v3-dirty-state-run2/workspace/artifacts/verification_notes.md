## Checked directly
- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code
- `src/release_preview/cli.py` defines the active subcommand as `generate`, exposes `--config` in live help, and resolves `RELEASE_PREVIEW_CONFIG` before the deprecated `--settings` and `PREVIEW_SETTINGS_PATH` names.
- `scripts/release_preview_helper.py` labels `build-preview` as `Deprecated compatibility alias` and prints a compatibility forward target to `python src/release_preview/cli.py generate --config configs/release_preview.toml`, so the helper path remains supported but is not primary.
- Conflicting prose in `docs/runbooks/release_preview.md` and `README_fragments/legacy_path.md` described the helper path as a peer transition path; that prose was overruled by bundle-local code and live help, which identify the helper alias and `--settings` path as deprecated compatibility only.

## Remaining caveats
- The helper alias still exists for backwards compatibility, so older automation may continue to function even though the runbook should direct operators to the current CLI.
- `pytest -q tests/test_release_preview_cli.py` passed, but emitted a cache write warning for `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/.pytest_cache/...` because that parent path is outside this writable bundle.
