# Verification Notes

Bundle-local files reviewed:

- `src/release_preview/cli.py`
- `scripts/release_preview_helper.py`
- `tests/test_release_preview_cli.py`
- `configs/release_preview.toml`

Exact checked commands:

1. `python src/release_preview/cli.py --help`
   Result: help lists the `generate` subcommand.
2. `python src/release_preview/cli.py generate --help`
   Result: help lists `--config` and `--dry-run`; it does not list `--settings`.
3. `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
   Result:
   - `entrypoint=python src/release_preview/cli.py generate`
   - `config=configs/release_preview.toml`
   - `mode=dry-run`
4. `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
   Result:
   - `deprecated_alias=true`
   - `compatibility_forward_to=python src/release_preview/cli.py generate --config configs/release_preview.toml`
5. `pytest -q tests/test_release_preview_cli.py`
   Result:
   - `2 passed`
   - Pytest emitted a non-fatal cache warning because it could not write `.pytest_cache` outside the writable sandbox.

Reconciliation summary:

- Primary runbook path is `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- Primary environment variable is `RELEASE_PREVIEW_CONFIG`.
- Deprecated compatibility inputs `--settings` and `PREVIEW_SETTINGS_PATH` still resolve configuration, but they are not the primary operator path.
- The helper alias still exists only to forward operators toward the current command.
