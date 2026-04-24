# Review Findings

## Changing the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects a machine-readable JSON object with a top-level version field. Switching the parser default to markdown means the existing no-flag invocation now emits human-readable Markdown instead of JSON, so downstream parsers will break immediately.
- missing test: Visible CI only exercised the markdown test file while the JSON contract suite remained skipped, so there is no coverage on the no-flag CLI path.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, review/ci_snapshot.md

## The json renderer registry now routes explicit json exports to Markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: The unchanged json renderer still defines the automation-facing payload shape, but the registry now maps the json key to render_markdown. As a result, export_report(..., output="json") and the CLI's --output json path return a Markdown string instead of the expected dictionary-like object, which changes the explicit JSON contract the PR said would stay intact.
- missing test: The stack did not run a json-contract test, so this regression is not covered by the visible CI that only passed on the markdown path.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md
