# Catalog Sync Preview Contract

The catalog-sync preview endpoint emits the normalized slug and route
bucket that sync workers currently use.

## Preview shape

`preview()` returns:

- `slug`
- `route_bucket`
- `route`

`route` uses the legacy bucket format:

`<route_bucket>:<slug>`

## Title normalization

The normalized title slug:

- trims surrounding whitespace
- lowercases the title before routing
- keeps the route bucket stable for a given owner and region

Examples:

- `Missing SKU` -> `missing-sku`
- ` Missing SKU ` -> `missing-sku`
