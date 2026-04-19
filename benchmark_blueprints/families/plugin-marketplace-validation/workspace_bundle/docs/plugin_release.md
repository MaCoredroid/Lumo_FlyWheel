# Release Sync Plugin

Install the plugin with the canonical marketplace settings below:

```toml
plugin_id = "release-sync"
startup_command = "python -m release_sync_mcp"
bundle = "dist/release-sync.zip"
```

The packaged zip is `dist/release-sync.zip`.

Release note: this update aligns the schema-v2 manifest, the module entrypoint, the canonical slug, the packaged icon path, and the marketplace tool id so the bundle metadata is consistent end to end.
