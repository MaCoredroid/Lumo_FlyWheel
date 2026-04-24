# Review Findings

## Changing the CLI default to markdown breaks existing no-flag automation
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects JSON with a top-level `version` field. With this change, `release-readiness <input>` now defaults to markdown instead, so existing consumers lose their machine-readable payload unless they are updated in lockstep.
- missing test: Visible CI only exercised `repo/tests/test_markdown_export.py`, while the JSON contract suite is still skipped in this stack.
- evidence: repo/src/release_readiness/cli.py, repo/docs/export_contract.md, review/pr_description.md, review/ci_snapshot.md

## The renderer registry now makes `--output json` return markdown content
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged JSON contract notes require explicit `--output json` to keep returning a dictionary-like payload. After this mapping change, `export_report(..., output="json")` returns the markdown string and the CLI prints a quoted markdown blob rather than an object with keys like `version`, which will break downstream parsers even when they pass `--output json` explicitly.
- missing test: The PR adds markdown snapshot coverage, but there is no active test in this stack that still exercises the explicit JSON path.
- evidence: repo/src/release_readiness/renderers/registry.py, repo/tests/test_json_contract_notes.md, repo/src/release_readiness/renderers/json_renderer.py, review/ci_snapshot.md
