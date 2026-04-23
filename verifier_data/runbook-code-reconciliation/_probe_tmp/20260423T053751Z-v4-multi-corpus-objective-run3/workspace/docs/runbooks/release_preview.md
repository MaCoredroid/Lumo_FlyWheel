# Daily Release Preview Runbook

Use the bundle's primary entrypoint:

```bash
python src/release_preview/cli.py generate --config configs/release_preview.toml
```

You can also provide the config path through the current environment variable:

```bash
export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml
python src/release_preview/cli.py generate
```

Deprecated compatibility path:

```bash
python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml
```

Treat the helper path, `--settings`, and `PREVIEW_SETTINGS_PATH` as legacy-only compatibility surfaces. They still exist for backwards compatibility, but the code, live help, and tests make `python src/release_preview/cli.py generate` with `--config` or `RELEASE_PREVIEW_CONFIG` the current operator path.
