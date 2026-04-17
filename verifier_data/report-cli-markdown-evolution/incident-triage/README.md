`incident-triage` is the richer incident-operations benchmark in the
`report-cli-markdown-evolution` family.

The visible task is still "add Markdown output and update the docs", but the
hidden bundle is built around a distinct latent defect: the shared ack-SLA
classifier treats threshold-equal incidents as healthy even though the runbook
defines the threshold as already breached.

The verifier bundle therefore checks:

- round-1 Markdown behavior and JSON stability on non-boundary examples
- differential parity against authored Markdown fixtures
- shortcut resistance against hardcoded and markdown-only workarounds
- follow-up coverage that forces the fix into the shared incident model and
  summary layer instead of a renderer-local workaround
