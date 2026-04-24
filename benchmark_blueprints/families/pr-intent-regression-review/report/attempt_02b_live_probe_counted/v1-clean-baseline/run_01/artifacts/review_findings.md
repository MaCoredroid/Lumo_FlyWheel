# Review Findings

## Switching the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects a machine-readable JSON object with a top-level `version` field. After this default change, `release-readiness <input>` emits markdown headings and bullets instead, so those consumers lose the JSON payload the PR description promised to preserve.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md

## The renderer registry makes `--output json` return markdown text
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: Unchanged contract notes still require explicit `--output json` to produce a dictionary-like payload, but the registry now maps `json` to `render_markdown`. That makes `export_report(..., output="json")` return a string, and the existing CLI path wraps it with `json.dumps(...)`, yielding a quoted markdown blob instead of the JSON object downstream parsers expect.
- evidence: review/pr_description.md, review/patch.diff, repo/tests/test_json_contract_notes.md
