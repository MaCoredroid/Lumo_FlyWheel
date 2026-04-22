# Worker A Notes

## Hunk Inventory

- `A1` `src/watchlist_report/cli.py`: plumb `--include-watchlist` through the markdown code path.
- `A2` `src/watchlist_report/renderers/markdown_renderer.py`: add summary bullets and a watchlist follow-up section.
- `A3` `src/watchlist_report/service.py`: rename JSON keys to `entries` and `watchlist_follow_up` for markdown consistency.
- `A4` `src/watchlist_report/renderers/json_renderer.py`: normalize the renamed keys during serialization.

## Risk Callout

`A3` and `A4` were written under the assumption that the JSON contract could
shift during the markdown salvage. Verify that assumption before keeping them.
