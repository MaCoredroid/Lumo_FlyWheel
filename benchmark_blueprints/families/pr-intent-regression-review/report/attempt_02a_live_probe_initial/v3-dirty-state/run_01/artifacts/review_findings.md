# Review Findings

## Default CLI output no longer preserves the automation-facing JSON contract
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: Changing `--output` to default to `markdown` means a bare `release-readiness <input>` invocation now emits the human markdown view instead of the machine-readable JSON object the unchanged export contract says nightly automation consumes without flags. That is a behavioral regression on the existing CLI surface, not just a documentation tweak.
- missing test: The CI bundle only exercised `repo/tests/test_markdown_export.py`, while the JSON contract suite was explicitly skipped in this stack.
- evidence: review/pr_description.md, review/patch.diff, review/ci_snapshot.md, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## Explicit `--output json` now returns markdown text instead of the JSON payload
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/export.py
- impact: The registry now maps the `json` key to `render_markdown`, so the unchanged `export_report(..., output='json')` call path returns a string rather than the dictionary produced by `render_json`. Any explicit JSON consumer now gets markdown content, and the unchanged CLI branch at `if args.output == "json": print(json.dumps(rendered, sort_keys=True))` will serialize that markdown string as a quoted JSON string instead of the historical top-level object.
- missing test: The new tests add markdown snapshot coverage only; nothing in this stack exercises `output='json'`, and the CI snapshot shows the JSON contract suite skipped.
- evidence: review/pr_description.md, review/patch.diff, review/ci_snapshot.md, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/export.py, repo/src/release_readiness/renderers/json_renderer.py, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md
