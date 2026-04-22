# Comparison Summary

Chosen strategy: Candidate B, the service-owned normalization fix.

Candidate B is the coherent repair because `service.compile_filters(...)`
is the shared contract boundary for the CLI, scheduled importer, and
saved-view repair job. Moving normalization ownership there fixes all
known callers with one implementation and keeps `cli.py` as a thin
parser.

Candidate A is weaker because it only patches `cli.py`. It makes the
visible CLI regression pass, but it leaves direct service callers
returning unnormalized keys such as `ops---latency__summary`. The
archived CLI-only memo is stale evidence because it predates the later
rollout of the importer and saved-view repair jobs onto
`service.compile_filters(...)`.
