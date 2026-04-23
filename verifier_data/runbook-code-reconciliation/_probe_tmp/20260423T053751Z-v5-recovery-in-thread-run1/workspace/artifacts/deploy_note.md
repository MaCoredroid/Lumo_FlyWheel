Deploy with the primary entrypoint:

`python src/release_preview/cli.py generate --config configs/release_preview.toml`

If you need environment-based configuration, use `RELEASE_PREVIEW_CONFIG`. The CLI code still honors deprecated fallbacks, but live help only exposes `--config` on the primary command.

`python scripts/release_preview_helper.py build-preview` remains available for backward compatibility, using `--settings`, but the helper identifies itself as a deprecated alias and forwards operators back to the primary CLI path. Do not treat it as the default release-preview workflow.
