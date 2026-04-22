# Comparison Summary

- Chosen Strategy: `candidate_b_service_layer`
- Rejected Strategy: `candidate_a_cli_local`

Candidate A is weaker because it only repairs the visible CLI path and
leaves direct callers of `service.compile_filters(...)` broken.

Candidate B wins because one shared service-layer owner fixes the CLI,
scheduled importer, and saved-view repair job without duplicating
normalization in `cli.py`.

Variant-specific evidence: The direct caller matrix makes the shared service layer the correct owner.
Required cues surfaced in this summary: service layer, direct caller, scheduled importer
