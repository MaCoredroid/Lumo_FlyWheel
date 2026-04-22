# Comparison Summary

Chosen strategy: Candidate B, the service-layer normalization fix.

Candidate B is stronger because `compile_filters(...)` is the shared boundary used by the CLI and by direct callers such as scheduled importers and repair jobs. Moving normalization ownership into `src/report_filters/service.py` makes every caller receive canonical labels, while keeping `src/report_filters/cli.py` as a thin parser.

Rejected strategy: Candidate A.

Candidate A is weaker because it only normalizes labels in the CLI path. The isolated evaluation showed that the CLI tests passed, but a direct `compile_filters(...)` call still returned `['ops---latency__summary', 'api__errors']`. That leaves service consumers inconsistent and preserves the underlying normalization bug for non-CLI callers.
