# Deploy Note

Reconciled on 2026-04-22 against bundle-local `src/release_preview/cli.py`, `scripts/release_preview_helper.py`, and `tests/test_release_preview_cli.py`.

Operator-facing guidance:

- Primary path: `python src/release_preview/cli.py generate --config configs/release_preview.toml`
- Preferred environment variable: `RELEASE_PREVIEW_CONFIG`
- Default config when no flag or env is supplied: `configs/release_preview.toml`
- Compatibility-only path retained: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

The helper alias remains available for older callers, but the runbook should direct humans to the bundle-local CLI, not the alias.
