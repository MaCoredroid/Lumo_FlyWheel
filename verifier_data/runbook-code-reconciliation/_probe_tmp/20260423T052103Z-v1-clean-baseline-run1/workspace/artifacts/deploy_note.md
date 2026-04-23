# Deploy Note

Release-preview operations for this bundle should use the current CLI directly:

```bash
python src/release_preview/cli.py generate --config configs/release_preview.toml
```

If operators prefer environment-based configuration, use:

```bash
export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml
python src/release_preview/cli.py generate
```

Compatibility-only alias:

```bash
python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
```

The compatibility alias should remain documented only as a fallback reference. It must not be presented as the primary release-preview procedure.
