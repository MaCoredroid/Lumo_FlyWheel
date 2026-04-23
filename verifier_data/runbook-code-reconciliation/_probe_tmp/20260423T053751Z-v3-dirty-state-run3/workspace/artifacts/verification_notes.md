## Checked directly
- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code
- `src/release_preview/cli.py` is the primary path. Live help exposes only the `generate` subcommand on `python src/release_preview/cli.py`, and `main()` prints `entrypoint=python src/release_preview/cli.py generate`.
- The current flag is `--config`; the deprecated flag is `--settings`. `generate --help` shows `--config` and suppresses `--settings`, while `resolve_config()` still accepts the deprecated name for compatibility.
- The current environment variable is `RELEASE_PREVIEW_CONFIG`; the deprecated compatibility variable is `PREVIEW_SETTINGS_PATH`.
- The helper path `python scripts/release_preview_helper.py build-preview` is legacy compatibility only. The helper code labels `build-preview` as a deprecated alias and prints `deprecated_alias=true` plus a forward target to the current CLI command.
- Conflicting prose was overruled by code and live help. The prior runbook text said to "use either" the current CLI path or the helper path during transition and told operators to export `PREVIEW_SETTINGS_PATH`; bundle-local code and live help show the current operator path is the `generate` subcommand with `--config`, while the helper and `PREVIEW_SETTINGS_PATH` remain deprecated compatibility surfaces.

## Remaining caveats
- The helper alias still exists and may appear in older automation, but it should not be documented as a peer operator path.
- `PREVIEW_SETTINGS_PATH` still resolves in code for backward compatibility, so removing it from operator guidance does not remove runtime support.
