# Deploy Note

Reconciled the release-preview operator guidance to the bundle-local CLI surface.

Primary path: `python src/release_preview/cli.py generate` with `--config` or `RELEASE_PREVIEW_CONFIG`.

Deprecated compatibility path: `python scripts/release_preview_helper.py build-preview` with `--settings`. This alias still exists, but bundle-local code and live help mark it as compatibility-only and forward users toward the current CLI.

Corrected the runbook to stop presenting both paths as equal and to stop recommending `PREVIEW_SETTINGS_PATH` as the preferred operator-facing configuration name.
