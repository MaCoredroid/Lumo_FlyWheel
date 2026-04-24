# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects JSON with a top-level version field. Switching the parser default to markdown means those callers now receive a human-readable markdown document instead of the machine-readable payload they depend on.
- missing test: The CI snapshot only exercised markdown tests while the downstream JSON integration suite remained skipped, so this default-output regression is currently unguarded.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md

## The json renderer registry entry now returns markdown for explicit --output json
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The unchanged CLI json branch still serializes whatever export_report returns as JSON. After remapping the json renderer to render_markdown, explicit --output json produces a JSON string containing markdown text instead of the historical dictionary-like payload, breaking downstream consumers that parse object fields such as version.
- missing test: The skipped json integration coverage in this stack means the type/shape regression on the explicit json path is not caught by the visible test suite.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
