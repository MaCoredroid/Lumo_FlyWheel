# Verification Notes

Bundle variant: `v1-clean-baseline`

The runbook was reconciled against bundle-local code in `src/release_preview/cli.py` and `scripts/release_preview_helper.py`, plus live bundle-local CLI help output.

## Exact checked commands

```bash
python src/release_preview/cli.py --help
python src/release_preview/cli.py generate --help
python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run
python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
pytest -q tests/test_release_preview_cli.py
```

## Findings

- The current primary entrypoint is `python src/release_preview/cli.py generate`.
- The current flag is `--config`; `generate --help` does not expose `--settings`.
- The current environment variable is `RELEASE_PREVIEW_CONFIG`.
- The deprecated environment variable `PREVIEW_SETTINGS_PATH` is still accepted in code, but it is not the preferred runbook path.
- The helper alias `build-preview` still exists for compatibility and prints a forward to `python src/release_preview/cli.py generate --config configs/release_preview.toml`.
- The bundle default config path remains `configs/release_preview.toml`.
- `configs/release_preview.toml` exists in the bundle and the dry-run command resolves it successfully.
