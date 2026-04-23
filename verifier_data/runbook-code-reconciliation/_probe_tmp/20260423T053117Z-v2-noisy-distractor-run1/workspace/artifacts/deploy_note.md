Primary release-preview invocation is `python src/release_preview/cli.py generate`.

Use `--config` for an explicit config path, or `RELEASE_PREVIEW_CONFIG` if the deploy environment injects configuration through the environment. The helper wrapper `python scripts/release_preview_helper.py build-preview` and the legacy names `--settings` / `PREVIEW_SETTINGS_PATH` remain compatibility-only and should not be used as the default deployment path.
