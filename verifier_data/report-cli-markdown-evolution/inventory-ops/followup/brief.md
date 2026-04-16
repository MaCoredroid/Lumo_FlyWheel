Round 2 follow-up for `inventory-ops`:

The morning handoff wants the on-shift watchlist owners to stay visible in the
Markdown owner totals even when they currently have zero queued items. The
repo already carries that intent in `docs/handoff.md` and in the dormant
`include_known_owners` flag.

Fix this at the shared summary layer, not with a Markdown-only workaround, and
make sure the shared wording uses the plural form for zero counts.
