# Worker B Notes

## Hunk Inventory

- `B1` `src/watchlist_report/renderers/markdown_renderer.py`: cleaner markdown headings and summary framing.
- `B2` `docs/usage.md`: add markdown CLI usage examples.
- `B3` `tests/fixtures/legacy_snapshot.md`: refresh a legacy snapshot while editing docs.

## Risk Callout

`B3` touches an unrelated fixture. Preserve the markdown/docs improvements only
if they can land without that snapshot churn.
