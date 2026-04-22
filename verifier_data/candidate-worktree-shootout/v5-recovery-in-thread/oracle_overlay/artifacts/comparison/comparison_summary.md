# Comparison Summary

- Chosen Strategy: `candidate_b_service_layer`
- Rejected Strategy: `candidate_a_cli_local`

Candidate A is weaker because it only repairs the visible CLI path and
leaves direct callers of `service.compile_filters(...)` broken.

Candidate B wins because one shared service-layer owner fixes the CLI,
scheduled importer, and saved-view repair job without duplicating
normalization in `cli.py`.

Variant-specific evidence: The rollback incident shows the CLI-local hotfix already failed in production and can not be reselected blindly.
Required cues surfaced in this summary: rollback, incident, incident_context/rollback_2026_07.md
