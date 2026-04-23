# Daily Release Preview Runbook

Use the current release-preview CLI entrypoint:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

If you prefer an environment variable over an explicit flag, export the current variable and run the same entrypoint:

`export RELEASE_PREVIEW_CONFIG=configs/release_preview.toml`

`python src/release_preview/cli.py generate`

Deprecated compatibility path:

`python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`

The helper still exists for backwards compatibility, but bundle-local code and live help identify it as a deprecated alias. The current operator-facing flag is `--config`. The current environment variable is `RELEASE_PREVIEW_CONFIG`. The deprecated names `--settings` and `PREVIEW_SETTINGS_PATH` remain fallback inputs in code and should not be treated as the primary runbook path.
