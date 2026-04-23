# Verification Notes

Date: 2026-04-22

Bundle-local sources checked:

- `src/release_preview/cli.py`
- `scripts/release_preview_helper.py`
- `tests/test_release_preview_cli.py`

Exact commands checked:

```bash
python3 src/release_preview/cli.py --help
python3 src/release_preview/cli.py generate --help
python3 src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run
python3 src/release_preview/cli.py generate --dry-run
RELEASE_PREVIEW_CONFIG=env/current.toml python3 src/release_preview/cli.py generate --dry-run
PREVIEW_SETTINGS_PATH=env/deprecated.toml python3 src/release_preview/cli.py generate --dry-run
python3 scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
pytest -q tests/test_release_preview_cli.py
```

Observed results summary:

- `python3 src/release_preview/cli.py --help` shows `generate` as the current subcommand.
- `python3 src/release_preview/cli.py generate --help` shows `--config` and `--dry-run`; it does not expose `--settings`.
- `python3 src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run` prints:
  - `entrypoint=python src/release_preview/cli.py generate`
  - `config=configs/release_preview.toml`
  - `mode=dry-run`
- `python3 src/release_preview/cli.py generate --dry-run` uses the default config `configs/release_preview.toml`.
- `RELEASE_PREVIEW_CONFIG=env/current.toml ... generate --dry-run` resolves to `config=env/current.toml`.
- `PREVIEW_SETTINGS_PATH=env/deprecated.toml ... generate --dry-run` resolves to `config=env/deprecated.toml`.
- `python3 scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` prints `deprecated_alias=true` and forwards to `python src/release_preview/cli.py generate --config configs/release_preview.toml`.

Reconciliation outcome:

- The runbook now uses `python src/release_preview/cli.py generate --config configs/release_preview.toml` as the primary operator command.
- The deprecated helper alias and deprecated env/flag names are documented as compatibility-only, not as the primary path.
