# Comparison Summary

- Chosen Strategy: `candidate_b_service_layer`
- Rejected Strategy: `candidate_a_cli_local`

Candidate A is weaker because it only repairs the visible CLI path and
leaves direct callers of `service.compile_filters(...)` broken.

Candidate B wins because one shared service-layer owner fixes the CLI,
scheduled importer, and saved-view repair job without duplicating
normalization in `cli.py`.

Variant-specific evidence: Release context makes the batch importer the current blocker, so the fix must serve direct callers.
Required cues surfaced in this summary: release blocker, batch importer, release_context/importer_callers.md
