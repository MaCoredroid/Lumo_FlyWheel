# Verification Notes

Bundle-local code reviewed:

- `src/release_preview/cli.py`
- `scripts/release_preview_helper.py`
- `tests/test_release_preview_cli.py`

Exact checked commands:

```bash
python src/release_preview/cli.py --help
python src/release_preview/cli.py generate --help
python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run
RELEASE_PREVIEW_CONFIG=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run
python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
pytest -q tests/test_release_preview_cli.py
```

Observed results before test execution:

- `python src/release_preview/cli.py --help` exposes the `generate` subcommand.
- `python src/release_preview/cli.py generate --help` exposes `--config` and `--dry-run`, and does not expose `--settings`.
- `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run` prints:
  - `entrypoint=python src/release_preview/cli.py generate`
  - `config=configs/release_preview.toml`
  - `mode=dry-run`
- `RELEASE_PREVIEW_CONFIG=configs/release_preview.toml python src/release_preview/cli.py generate --dry-run` resolves the same config path.
- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml` prints:
  - `deprecated_alias=true`
  - `compatibility_forward_to=python src/release_preview/cli.py generate --config configs/release_preview.toml`

Reconciliation outcome:

- The primary runbook path is the bundle-local CLI: `python src/release_preview/cli.py generate --config configs/release_preview.toml`
- The helper alias remains compatibility-only and is not the primary documented path.
- The current env var is `RELEASE_PREVIEW_CONFIG`; `PREVIEW_SETTINGS_PATH` is deprecated compatibility behavior.

Required test result:

- `pytest -q tests/test_release_preview_cli.py` passed with `2 passed, 1 warning in 0.07s`
- The warning was a non-fatal `PytestCacheWarning` caused by an unwritable `.pytest_cache` path outside the editable bundle root.
