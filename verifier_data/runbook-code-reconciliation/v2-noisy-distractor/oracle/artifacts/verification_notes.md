# Verification Notes

## Checked directly
- `python src/release_preview/cli.py --help` proves the live entrypoint is `python src/release_preview/cli.py generate`.
- `python src/release_preview/cli.py generate --help` proves the current visible flag is `--config`.
- `python scripts/release_preview_helper.py build-preview --help` proves the legacy alias still exists for compatibility.
- `pytest -q tests/test_release_preview_cli.py` confirms the bundle-local CLI contract still passes without editing code.

## Inferred from code
- `src/release_preview/cli.py` prefers `RELEASE_PREVIEW_CONFIG` and only falls back to `PREVIEW_SETTINGS_PATH`.
- `.env.example` documents `RELEASE_PREVIEW_CONFIG` as current and marks `PREVIEW_SETTINGS_PATH` as deprecated.

## Remaining caveats
- The compatibility helper still works, so stale prose can look partially correct even though the runbook should prefer the current CLI path.
- README fragments disagree with one another; code and live help remain the source of truth.
