# Review Findings

## Switching the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects JSON with a top-level `version` field. Flipping the parser default to `markdown` means those callers now receive a human-readable markdown document instead of the machine-readable object the PR description promised to keep intact.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md

## The registry now returns markdown for explicit json exports
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: Unchanged callers still rely on `export_report(..., output="json")` and `--output json` producing the historical dictionary payload, but the registry now maps the `json` key to `render_markdown`. As a result the CLI JSON path serializes a markdown string, which removes the object shape automation consumers depend on.
- evidence: review/pr_description.md, review/patch.diff, repo/tests/test_json_contract_notes.md
