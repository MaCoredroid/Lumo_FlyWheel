# Review Findings

## Changing the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects machine-readable JSON with a top-level version field. Flipping the parser default to markdown means the same no-flag invocation now prints human-readable markdown instead, so existing automation consumers stop receiving the JSON payload the PR description promised to preserve.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## The json renderer registry entry now routes explicit json requests to markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The unchanged CLI json branch still assumes export_report(..., output='json') returns a dictionary-like payload and serializes it with json.dumps. After this registry change, explicit --output json resolves to render_markdown, so callers get a quoted markdown string instead of the historical object shape, violating the documented byte-for-byte json compatibility guarantee for downstream consumers.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, review/ci_snapshot.md
