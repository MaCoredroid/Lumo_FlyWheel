# Candidate B

Strategy: move normalization ownership to
`src/report_filters/service.py`, keep `cli.py` thin, and add a service
regression test.

Strength:
- fixes CLI and direct service callers with one shared contract

Risk:
- must keep docs aligned so future callers do not reintroduce CLI-local
  normalization
