## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` sets the parser program to `release-preview`, exposes `generate` as the active subcommand, and prints `entrypoint=python src/release_preview/cli.py generate` from `main()`. That overrules the stale README fragments and runbook prose that centered the helper alias.
- `src/release_preview/cli.py` registers `--config` normally but suppresses `--settings` from `generate --help`, so `--config` is the current flag and `--settings` is a deprecated compatibility flag even though it still parses.
- `resolve_config()` prefers `args.config`, then `RELEASE_PREVIEW_CONFIG`, then deprecated `args.deprecated_settings`, then deprecated `PREVIEW_SETTINGS_PATH`, then the default config path. This overrules the old prose that told operators to export `PREVIEW_SETTINGS_PATH` as the standard path.
- `scripts/release_preview_helper.py` labels `build-preview` as a deprecated compatibility alias and forwards to `python src/release_preview/cli.py generate --config configs/release_preview.toml`, so backwards compatibility does not make the helper the primary entrypoint.

## Remaining caveats

- The deprecated helper alias is still callable and still shows `--settings` in its own help output, so older notes may appear plausible even though they no longer describe the primary path.
- The current CLI help confirms the current command and visible flag, but environment-variable precedence is only visible by reading code.
- The required pytest run passed, but pytest emitted a cache-write warning for `/Users/zhiyuanma/work/CursorWS/Lumo_FlyWheel/.pytest_cache` outside this writable workspace.
