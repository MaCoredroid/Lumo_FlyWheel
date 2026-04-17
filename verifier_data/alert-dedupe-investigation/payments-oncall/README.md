Rich verifier bundle for `alert-dedupe-investigation/payments-oncall`.

The visible repo fails because the payments dedupe path stopped normalizing
environment aliases and 5-minute windows before grouping alert bursts. The rich
bundle extends that benchmark with payments-specific follow-up coverage:

- `payment_lane` remains part of the incident identity
- title-only noise should not split an incident when the stream provides a
  stable `dedupe_hint`
- the grouped handoff keeps the earliest human-readable title even when later
  alerts add processor-specific display suffixes
