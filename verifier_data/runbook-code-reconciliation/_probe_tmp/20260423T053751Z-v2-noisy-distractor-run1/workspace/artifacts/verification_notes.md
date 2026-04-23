## Checked directly

- `python src/release_preview/cli.py --help`
- `python src/release_preview/cli.py generate --help`
- `python scripts/release_preview_helper.py build-preview --help`
- `python src/release_preview/cli.py generate --dry-run`
- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run`
- `PREVIEW_SETTINGS_PATH=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run`
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- `pytest -q tests/test_release_preview_cli.py`

## Inferred from code

- `src/release_preview/cli.py` sets `prog="release-preview"` and exposes `generate` as the active workflow, so the primary entrypoint is the current CLI at `python src/release_preview/cli.py`, not the helper wrapper.
- `src/release_preview/cli.py` defines `CURRENT_FLAG = "--config"` and suppresses help for `--settings`, so live help overruled the older runbook text that treated `--settings` as the normal operator flag.
- `src/release_preview/cli.py` resolves configuration in this order: `--config`, `RELEASE_PREVIEW_CONFIG`, deprecated `--settings`, deprecated `PREVIEW_SETTINGS_PATH`, then the bundled default config. That makes `RELEASE_PREVIEW_CONFIG` the current env var and `PREVIEW_SETTINGS_PATH` a fallback only.
- `scripts/release_preview_helper.py` labels `build-preview` as a deprecated compatibility alias and prints a forward target to the current CLI command. The helper therefore remains supported for compatibility, but it is not the primary path.
- `README_fragments/current_path.md` correctly names the current entrypoint, but its retained `PREVIEW_SETTINGS_PATH` example was overruled by code and live help. `README_fragments/legacy_path.md` is consistent with a legacy compatibility path, not the current operator instruction.

## Remaining caveats

- The helper alias still accepts `--settings`, so older automation may continue to work even though the operator runbook should prefer the current CLI.
- The current CLI still honors deprecated names through fallback resolution, so compatibility should not be mistaken for primary status.
