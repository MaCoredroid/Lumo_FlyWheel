# Architecture

The release-readiness CLI is organized in layers. Each layer has a narrow
responsibility and depends only on layers below it.

```
cli.py                         # argparse, entry point
  │
  ├── config.py                # pydantic-settings
  │
  ├── adapters/                # I/O boundary: read raw records
  │   ├── env_source.py
  │   └── fs_source.py
  │
  ├── core/                    # pure domain logic, no I/O
  │   ├── model.py             # dataclasses: Section, OwnerTotal, Report
  │   ├── aggregation.py       # raw records -> Report
  │   └── formatting.py        # shared string helpers (pluralization, tables)
  │
  └── renderers/               # Report -> string
      ├── base.py              # Renderer protocol
      ├── registry.py          # entry-point discovery
      └── json_renderer.py
```

## Design choices

- **Renderers are discovered via entry points.** This means any package
  installed in the environment can register a renderer without the core
  package knowing about it. The CLI reads the registry to populate
  `--format` choices, so an unregistered format is never exposed to users.
- **Formatting helpers are shared.** Every renderer that emits
  human-readable pluralization should call `core.formatting.format_count`
  so that "1 item" vs "2 items" is consistent across outputs.
- **The domain model is immutable.** `Section`, `OwnerTotal`, and `Report`
  are frozen dataclasses. Renderers cannot mutate reports.
