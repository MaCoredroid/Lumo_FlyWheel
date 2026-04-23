# Export Contract

The CLI keeps `json` as the default output because nightly automation shells out
without flags and expects a machine-readable object with a top-level `version`
field.

`--output markdown` is opt-in for humans. Explicit `--output json` must remain
byte-for-byte compatible with the existing automation-facing contract.
