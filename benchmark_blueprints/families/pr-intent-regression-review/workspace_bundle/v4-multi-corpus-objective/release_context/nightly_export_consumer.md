# Nightly Export Consumer

The nightly release-readiness fanout shells out to `release-readiness input.json`
without passing `--output`. It parses the default stdout as JSON and keys off
the top-level `version` field.
