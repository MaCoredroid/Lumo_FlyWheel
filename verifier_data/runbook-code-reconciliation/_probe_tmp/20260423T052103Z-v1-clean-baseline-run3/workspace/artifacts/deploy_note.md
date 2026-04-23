# Deploy Note

Release preview operations for `v1-clean-baseline` should be invoked through the current bundle-local CLI:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

`RELEASE_PREVIEW_CONFIG` is the current environment override. The helper alias and older compatibility inputs still exist, but they are fallback-only:

- helper alias: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- deprecated flag: `--settings`
- deprecated env var: `PREVIEW_SETTINGS_PATH`

For operator verification, use:

`python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run`
