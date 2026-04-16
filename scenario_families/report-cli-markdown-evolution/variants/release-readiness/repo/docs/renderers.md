# Adding a renderer

Renderers are discovered via Python entry points. The CLI reads the registry
to populate `--format` choices.

## Steps

1. Create a module under `src/release_readiness/renderers/` that defines a
   class implementing the `Renderer` protocol (see `renderers/base.py`).
2. Register the class in `pyproject.toml`:

   ```toml
   [project.entry-points."release_readiness.renderers"]
   <format-name> = "release_readiness.renderers.<module>:<ClassName>"
   ```

3. Reinstall the package so setuptools re-reads entry points:

   ```bash
   pip install -e .
   ```

4. Update this document with a brief note about the new renderer.

## Notes

- Renderers must use `core.formatting.format_count` for any pluralized
  counts so output is consistent across formats.
- Renderers are instantiated once per CLI invocation. They must not hold
  per-report state between `render()` calls.
- Renderers should be deterministic: the same `Report` input must always
  produce the same string output.

## Registered renderers

| Format | Module |
|---|---|
| `json` | `release_readiness.renderers.json_renderer:JsonRenderer` |
