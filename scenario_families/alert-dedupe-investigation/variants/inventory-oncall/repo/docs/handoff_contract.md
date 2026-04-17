## Inventory On-Call Handoff Contract

The alert stream keeps the raw page title for human readability, but inventory
dedupe must key incidents on the stable routing metadata instead of noisy
display text alone.

Normalize each raw event before collapsing:

- `environment`: aliases such as `Production` and `prod` canonicalize to `prod`
- `window_start`: derived from `observed_at` and floored into a 1-minute bucket
- `title`: preserve the display title on the grouped record

The dedupe fingerprint for a grouped incident is:

1. `service`
2. canonical `environment`
3. canonical 1-minute `window_start`
4. `inventory_scope`
5. stable incident family: prefer `dedupe_hint` when it is present, otherwise
   fall back to the normalized display title

Grouped incidents must preserve:

- the earliest display `title`
- the canonical `environment`
- the canonical `window_start`
- `inventory_scope`
- `dedupe_hint`
- `occurrence_count`
- `first_seen_at`
- `last_seen_at`
