Release-preview reconciliation is documentation-only. The primary operator path is `python src/release_preview/cli.py generate`, with `--config` and `RELEASE_PREVIEW_CONFIG` as the current configuration inputs.

`python scripts/release_preview_helper.py build-preview`, `--settings`, and `PREVIEW_SETTINGS_PATH` remain deprecated compatibility surfaces only. The runbook and reconciliation facts were updated to stop presenting them as equal alternatives.
