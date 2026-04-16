The project board sync flow needs owner routing wired through every layer.

Tests now expect the backend store, service API, CLI entrypoint, default config,
and docs to understand an `owner` field. Add the field end-to-end while keeping
the existing `name` and `status` behavior intact. If `--owner` is omitted,
the service should fall back to the default owner recorded in
`config/defaults.json`, and explicit owner values must override that default.
Downstream consumers also need to see whether the effective owner came from the
explicit flag or the default config, so the synced payload should expose an
`owner_source` field with `explicit` or `default`. The payload should also
include a canonical `routing_key` derived from the effective owner and a
normalized item slug so project board routes stay stable even when the item
name includes extra internal whitespace.

Do not remove the schema checks or the CLI coverage.
