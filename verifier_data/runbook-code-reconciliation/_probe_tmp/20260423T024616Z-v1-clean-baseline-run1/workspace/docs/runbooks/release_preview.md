# Daily Release Preview Runbook

Use the bundle-local CLI as the primary path for the daily release preview.

## Primary Command

1. Run:
   `python src/release_preview/cli.py generate --config configs/release_preview.toml`

2. For a non-writing verification pass, run:
   `python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`

## Config Resolution

The current CLI resolves configuration in this order:

1. `--config`
2. `RELEASE_PREVIEW_CONFIG`
3. Deprecated `--settings`
4. Deprecated `PREVIEW_SETTINGS_PATH`
5. Default `configs/release_preview.toml`

If you want to avoid passing the flag on every run, export:

`export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

Then run:

`python src/release_preview/cli.py generate`

## Compatibility Note

The helper alias still exists for compatibility:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

Do not use the helper alias as the primary runbook path. Its role is compatibility-only and it forwards operators back to the current CLI invocation.
