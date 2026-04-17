Rich verifier bundle for `alert-dedupe-investigation/inventory-oncall`.

The visible repo fails because the inventory dedupe path stopped normalizing
environment aliases and minute windows before grouping alert bursts. The rich
bundle extends that benchmark with inventory-specific follow-up coverage:

- `inventory_scope` remains part of the incident identity
- title-only batch noise should not split an incident when the stream provides a
  stable `dedupe_hint`
- the grouped handoff keeps the earliest human-readable title even when later
  alerts add batch-specific display suffixes
