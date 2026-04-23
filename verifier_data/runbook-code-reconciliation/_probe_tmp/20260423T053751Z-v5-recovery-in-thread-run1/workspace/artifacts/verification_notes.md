## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines `CURRENT_FLAG = "--config"` and `CURRENT_ENV = "RELEASE_PREVIEW_CONFIG"`, so those are the current names even though the CLI still accepts deprecated fallbacks internally.
- `src/release_preview/cli.py` also defines `DEPRECATED_FLAG = "--settings"` and `DEPRECATED_ENV = "PREVIEW_SETTINGS_PATH"`, but `generate --help` suppresses `--settings`, which is strong evidence that the old flag is compatibility-only.
- `scripts/release_preview_helper.py` labels `build-preview` as `Deprecated compatibility alias` and prints a forward target to `python src/release_preview/cli.py generate --config configs/release_preview.toml`, so the helper is not the primary path.
- Earlier runbook prose that presented both operator paths as equivalent was overruled by code and live help. The helper still exists, but only as a deprecated alias.
- README and incident prose saying the helper alias still exists for backward compatibility was retained as supporting evidence, but not as authority for default status. Code and live help decide that only the primary CLI path is current.

## Remaining caveats

- Running `python src/release_preview/cli.py --help` shows the top-level command and subcommand names, but not environment variable behavior; the env-var precedence comes from reading `resolve_config()` in code.
- The primary CLI still accepts deprecated names through code paths that are intentionally hidden from help, so older automation may continue to work even though operators should migrate.
