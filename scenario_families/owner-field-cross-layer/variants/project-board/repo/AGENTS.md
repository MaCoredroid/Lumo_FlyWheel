The project board sync flow needs owner routing wired through every layer.

Tests now expect the backend store, service API, CLI entrypoint, default config,
and docs to understand an `owner` field. Add the field end-to-end while keeping
the existing `name` and `status` behavior intact. If `--owner` is omitted,
the service should fall back to the default owner recorded in
`config/defaults.json`, and explicit owner values must override that default.

Do not remove the schema checks or the CLI coverage.
