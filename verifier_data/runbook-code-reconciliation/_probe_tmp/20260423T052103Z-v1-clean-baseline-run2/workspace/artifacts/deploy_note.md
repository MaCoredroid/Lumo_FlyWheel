# Deploy Note

Use the current release-preview entrypoint:

- `python src/release_preview/cli.py generate --config configs/release_preview.toml`

Use `RELEASE_PREVIEW_CONFIG` only when an environment-based override is needed.

Do not use the helper alias as the primary deploy/runbook path. The legacy helper remains available only for compatibility:

- `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Observed dry-run output for the current path:

- `entrypoint=python src/release_preview/cli.py generate`
- `config=configs/release_preview.toml`
- `mode=dry-run`
