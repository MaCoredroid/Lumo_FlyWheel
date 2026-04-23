## Checked directly
- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `python src/release_preview/cli.py generate --dry-run`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code
- `src/release_preview/cli.py` is the primary path because it owns the real `generate` subcommand and prints `entrypoint=python src/release_preview/cli.py generate` during execution.
- `scripts/release_preview_helper.py` is a deprecated compatibility alias because it prints `deprecated_alias=true` and forwards to the direct CLI command string.
- The current names come from code, not prose: `--config` and `RELEASE_PREVIEW_CONFIG` are current, while `--settings` and `PREVIEW_SETTINGS_PATH` are deprecated fallbacks.
- The original runbook prose treated both operator paths as equivalent and told operators to export `PREVIEW_SETTINGS_PATH`; that was overruled by the live CLI help and resolver order in code, which make the direct CLI and `RELEASE_PREVIEW_CONFIG` the current defaults.
- `tests/test_release_preview_cli.py` confirms the same split: `generate --help` exposes `--config` and suppresses `--settings`, while the helper alias is retained only for compatibility.

## Remaining caveats
- The helper help output still shows `--settings`, so operators can continue using it during transition, but the bundle-local code marks it as compatibility only.
- No live deployment was performed here; reconciliation is limited to code, CLI help, and the targeted test suite.
