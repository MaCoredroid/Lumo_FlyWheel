# Alert Preview Contract

The alert-routing preview endpoint emits the normalized slug and route
bucket that on-call routing currently uses.

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

- `Disk Pressure` -> `disk-pressure`
- ` Disk Pressure ` -> `disk-pressure`
