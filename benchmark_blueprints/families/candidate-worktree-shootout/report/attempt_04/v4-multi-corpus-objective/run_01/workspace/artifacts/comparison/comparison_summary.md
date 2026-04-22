# Comparison Summary

Chosen strategy: Candidate B, which moves separator-heavy label
normalization into `service.compile_filters(...)` and keeps the CLI as a
thin pass-through.

Candidate B is stronger because the scheduled importer and saved-view
repair job call the service layer directly. Fixing the shared service
contract resolves the immediate release blocker and keeps every caller on
one normalization path.

Candidate A is weaker because it only patches `cli.render_filters(...)`.
That makes the visible CLI tests pass, but it leaves direct
`service.compile_filters(...)` callers returning
`ops---latency__summary`, which preserves the importer regression.
