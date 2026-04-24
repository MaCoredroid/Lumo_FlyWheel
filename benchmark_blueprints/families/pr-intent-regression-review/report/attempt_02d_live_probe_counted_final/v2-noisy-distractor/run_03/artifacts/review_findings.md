# Review Findings

## Changing the parser default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged contract still says nightly automation shells out without flags and expects JSON by default. With `build_parser().parse_args(['input.json']).output` now resolving to `markdown`, the unchanged `main()` path prints the human markdown view for no-flag invocations instead of the machine-readable object those callers consume.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## The json renderer registry entry now routes explicit JSON requests to markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: The unchanged `render_json()` implementation still defines the automation-facing object shape, but the registry now maps `"json"` to `render_markdown`. As a result, `export_report(..., output='json')` returns a string, and the unchanged CLI `json.dumps(rendered, sort_keys=True)` branch emits a quoted markdown blob instead of the top-level JSON object promised to downstream consumers.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/renderers/json_renderer.py, repo/src/release_readiness/cli.py
