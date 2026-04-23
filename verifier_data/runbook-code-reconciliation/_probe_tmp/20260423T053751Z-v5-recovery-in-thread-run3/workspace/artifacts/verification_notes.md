## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python src/release_preview/cli.py generate --dry-run`
- `python scripts/release_preview_helper.py --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` is the primary path: the parser exposes `generate`, the generated runtime output reports `entrypoint=python src/release_preview/cli.py generate`, and `resolve_config()` prefers `--config` then `RELEASE_PREVIEW_CONFIG`.
- `scripts/release_preview_helper.py` is deprecated compatibility only: top-level help labels `build-preview` as `Deprecated compatibility alias`, and running it prints `deprecated_alias=true` plus a forward target to the current CLI command.
- Conflicting prose was overruled by code and live help. The prior runbook treated the helper path as an equal operator path and told operators to export `PREVIEW_SETTINGS_PATH` for helper usage; bundle-local code shows the current names are `--config` and `RELEASE_PREVIEW_CONFIG`, while `--settings` and `PREVIEW_SETTINGS_PATH` are deprecated fallback names rather than the preferred interface.

## Remaining caveats

- Direct verification covered help surfaces, alias behavior, and focused CLI tests; it did not run a full preview generation against real release data.
- `pytest -q tests/test_release_preview_cli.py` passed, but emitted a `PytestCacheWarning` because the sandbox could not write the parent repo `.pytest_cache` path.
- Deprecated names still work for backward compatibility, so callers may continue to succeed while using non-preferred interfaces; the runbook now treats that support as legacy-only.
