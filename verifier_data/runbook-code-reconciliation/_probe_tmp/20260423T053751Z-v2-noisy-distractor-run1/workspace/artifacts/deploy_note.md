Deploy note for `v2-noisy-distractor` release preview:

- Primary invocation: `python src/release_preview/cli.py generate --config configs/release_preview.toml`
- Current env override: `RELEASE_PREVIEW_CONFIG`
- Deprecated compatibility path: `python scripts/release_preview_helper.py build-preview --settings configs/release_preview.toml`
- Deprecated env fallback: `PREVIEW_SETTINGS_PATH`

Operational guidance:

- Update operator docs and automation to call the current CLI directly.
- Keep the helper path only where backward compatibility is still required.
