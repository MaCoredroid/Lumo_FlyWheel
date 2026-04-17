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
- collapses repeated whitespace
- treats separator noise from ledger exports such as `/`, `_`, and repeated
  punctuation as word breaks
- strips punctuation-only fragments
- never emits doubled or trailing hyphens

Examples:

- `Invoice Retry` -> `invoice-retry`
- `Refund / Retry _ Queue` -> `refund-retry-queue`
- `Chargeback---Retry!!!` -> `chargeback-retry`
- `Invoice 2024 / Retry 7` -> `invoice-2024-retry-7`
