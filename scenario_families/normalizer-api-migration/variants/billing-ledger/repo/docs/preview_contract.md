# Billing Preview Contract

The billing-ledger preview endpoint emits a canonical dispatch identity for
finance-ledger actions. The route bucket stays stable, while the dispatch
key carries the canonical slug that downstream reconciliation uses.

## Preview shape

`preview()` returns:

- `slug`
- `route_bucket`
- `dispatch_key`
- `route`

`route` keeps the legacy bucket format and appends the canonical dispatch key:

`<route_bucket>:<slug>?dispatch=<dispatch_key>`

## Canonical dispatch key

`dispatch_key` is:

`<region>:<owner>:<normalized-title-slug>`

The normalized title slug:

- lowercases all text
- collapses repeated whitespace into a canonical hyphenated slug
- stays stable even when upstream billing exports contain formatting noise
- never emits doubled or trailing hyphens

Examples:

- `Invoice Retry` -> `invoice-retry`
- ` Invoice   Retry ` -> `invoice-retry`
