# Daily Release Preview Runbook

This bundle's current CLI entrypoint is:

```bash
python src/release_preview/cli.py generate --config configs/release_preview.toml
```

## Daily operator path

1. Confirm the CLI surface if needed:

   ```bash
   python src/release_preview/cli.py generate --help
   ```

2. Run the preview with the explicit bundle-local config:

   ```bash
   python src/release_preview/cli.py generate --config configs/release_preview.toml
   ```

3. Optional dry-run check:

   ```bash
   python src/release_preview/cli.py generate --config configs/release_preview.toml --dry-run
   ```

## Config resolution

- Preferred flag: `--config`
- Preferred environment variable: `RELEASE_PREVIEW_CONFIG`
- Bundle default when no flag or env var is set: `configs/release_preview.toml`

Example environment-based invocation:

```bash
export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml
python src/release_preview/cli.py generate
```

## Compatibility note

The legacy helper alias still exists for compatibility:

```bash
python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
```

Do not use the helper alias as the primary runbook path. It only reports a compatibility forward to the current CLI command.
