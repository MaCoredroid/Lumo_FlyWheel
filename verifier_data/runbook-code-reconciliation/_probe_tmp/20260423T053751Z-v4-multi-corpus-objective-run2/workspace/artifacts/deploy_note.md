Reconciled the release-preview operator docs to the bundle-local CLI contract.

Primary path:
- `python src/release_preview/cli.py generate`
- current flag: `--config`
- current env var: `RELEASE_PREVIEW_CONFIG`

Deprecated compatibility path:
- `python scripts/release_preview_helper.py build-preview`
- deprecated flag: `--settings`
- deprecated env var: `PREVIEW_SETTINGS_PATH`

The earlier prose that treated both paths as equal was not kept. Bundle-local code and live help show the CLI under `src/release_preview/cli.py` as the primary interface, with the helper preserved only for backward compatibility.
