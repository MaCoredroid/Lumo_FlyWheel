## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --dry-run`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` defines `CURRENT_FLAG = "--config"` and `CURRENT_ENV = "RELEASE_PREVIEW_CONFIG"`, while `DEPRECATED_FLAG = "--settings"` and `DEPRECATED_ENV = "PREVIEW_SETTINGS_PATH"` are only fallback compatibility inputs during config resolution.
- `generate --help` exposes `--config` and hides `--settings`, so the live CLI surface treats `--config` as current even though the parser still accepts the deprecated flag internally.
- `scripts/release_preview_helper.py` labels `build-preview` as `Deprecated compatibility alias`, so the helper remains available but is not the preferred runbook path.
- Conflicting prose was overruled by code and live help: the existing runbook said to "use either" path, and supporting README/release-context prose noted that operators liked the helper, but the bundle-local code and CLI help make `python src/release_preview/cli.py generate` the primary path and the helper legacy-only.

## Remaining caveats

- The helper alias still works as a compatibility layer, so older operator habits may persist unless they are redirected to the primary CLI.
- Deprecated names are still honored as fallbacks in config resolution, which means drift can reappear if downstream docs copy compatibility examples instead of the primary command.
