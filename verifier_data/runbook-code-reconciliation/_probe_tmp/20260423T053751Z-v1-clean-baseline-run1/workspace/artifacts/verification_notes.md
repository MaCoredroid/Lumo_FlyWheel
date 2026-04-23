## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `pytest -q tests/test_release_preview_cli.py`

The live help confirms that the current operator-facing command is the `generate` subcommand on `python src/release_preview/cli.py`, with `--config` as the visible config flag. The helper path still exists, but only as `build-preview` on the separate helper script.

## Inferred from code

- `src/release_preview/cli.py` defines `CURRENT_FLAG = "--config"` and `CURRENT_ENV = "RELEASE_PREVIEW_CONFIG"`, then resolves those ahead of deprecated names in `resolve_config(...)`.
- The same file suppresses `--settings` from current CLI help via `help=argparse.SUPPRESS`, which matches the live help output that omits `--settings`.
- `scripts/release_preview_helper.py` names `build-preview` as a `Deprecated compatibility alias` and prints `deprecated_alias=true`, so it is not the primary operator path even though it still works.
- Conflicting README-style prose was overruled by code and live help: the existing runbook text and `README_fragments/legacy_path.md` pointed operators to `python scripts/release_preview_helper.py build-preview --settings ...` with `PREVIEW_SETTINGS_PATH`, but bundle-local code and live help show the primary path is `python src/release_preview/cli.py generate --config ...` with `RELEASE_PREVIEW_CONFIG`.

## Remaining caveats

- The current CLI help does not surface deprecated `--settings`, so its continued support is inferred from code and from the compatibility helper test rather than from current help text alone.
- `primary_entrypoint` and `legacy_alias` in the facts file are recorded as entrypoint commands only, per instruction, so subcommands and default config paths are documented in the runbook rather than embedded in those JSON fields.
- The pytest run passed, but emitted a cache-write warning outside this bundle workspace; that warning does not change the CLI reconciliation result.
