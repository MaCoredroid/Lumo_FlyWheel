
# Plugins

Built CLI usage comes from `node dist/src/index.mjs`.

The modernized loader resolves plugins from emitted `dist/plugins/`.

Supported module shapes:

- default-export plugin object
- named `plugin` export

Malformed modules fail with `Invalid plugin module: <name>`.

Regenerate `artifacts/cli_capture.txt` with `bash scripts/capture_cli_output.sh`.
