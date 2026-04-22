# Comparison Summary

Chosen strategy: Candidate B, the service-layer normalization fix.

Candidate B is stronger because `compile_filters(...)` is the shared
entry point for both CLI and non-CLI callers. Moving normalization
ownership into `src/report_filters/service.py` keeps `cli.py` thin and
ensures direct callers no longer emit separator-heavy keys.

Candidate A is weaker because it only repairs `src/report_filters/cli.py`.
Its isolated validation passed the existing suite, but direct service
usage still returned `ops---latency__summary` instead of the canonical
`ops latency summary`. That leaves the importer and saved-view repair
paths exposed whenever they bypass the CLI.
