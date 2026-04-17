Rich verifier bundle for `alert-dedupe-investigation/search-oncall`.

The visible repo fails because the search dedupe path stopped normalizing
environment aliases and minute windows before grouping alert bursts. The rich
bundle extends that benchmark with search-specific follow-up coverage:

- `search_cluster` remains part of the incident identity
- title-only shard noise should not split an incident when the stream provides a
  stable `dedupe_hint`
- the grouped handoff keeps the earliest human-readable title even when later
  alerts add shard-specific display suffixes
