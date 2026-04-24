# Review Findings

## Changing the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The PR description says existing JSON behavior must stay intact, but no-flag CLI invocations now take the markdown branch by default. The unchanged export contract says downstream automation shells out without flags and expects JSON with a top-level version field, so this silently changes the payload format for existing callers.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## The json renderer registry entry now returns markdown instead of a JSON payload
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: Mapping the json key to render_markdown means export_report(..., output="json") no longer returns the dictionary-like structure that the unchanged export API and CLI json branch expect. When callers pass --output json, the CLI now json.dumps() a markdown string and emits a quoted markdown blob instead of the machine-readable object promised to automation consumers.
- missing test: The visible CI only exercised markdown and skipped the json integration suite, so this contract regression was not covered in this stack.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, repo/src/release_readiness/export.py, repo/src/release_readiness/cli.py, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
