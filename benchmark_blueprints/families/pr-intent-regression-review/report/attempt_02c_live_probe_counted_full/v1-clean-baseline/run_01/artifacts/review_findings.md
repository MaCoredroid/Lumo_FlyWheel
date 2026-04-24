# Review Findings

## Changing the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and relies on JSON being the default. After this change, `release-readiness input.json` takes the markdown branch and prints a human-readable document instead of a machine-readable payload with a top-level `version`, so existing no-flag consumers will fail even though the PR description says existing JSON behavior should stay intact.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/docs/export_contract.md

## The json renderer registry entry now returns markdown for explicit `--output json`
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged JSON contract notes say explicit `--output json` must keep returning a dictionary-like payload. Routing the `json` key to `render_markdown` makes `export_report(..., output="json")` return a string, and the unchanged CLI then `json.dumps(...)` that string into a quoted markdown blob. Downstream callers asking for JSON lose the structured object and its `version` field entirely.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/src/release_readiness/renderers/registry.py, repo/tests/test_json_contract_notes.md
