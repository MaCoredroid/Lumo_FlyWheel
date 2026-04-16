# release-readiness

A small CLI that produces the daily release-readiness report for the release
management team.

## Quick start

```bash
make install
python -m release_readiness --format json
```

## Architecture

The CLI is thin. It reads configuration via `pydantic-settings`, loads
section records from a configured adapter (filesystem or environment), runs
them through the aggregation pipeline, and hands the resulting domain
objects to a registered renderer. Renderers are discovered via Python entry
points declared in `pyproject.toml` under `release_readiness.renderers`.

See `docs/architecture.md` for the layered design.
See `docs/renderers.md` for how to add a new output format.
See `docs/usage.md` for CLI usage.
