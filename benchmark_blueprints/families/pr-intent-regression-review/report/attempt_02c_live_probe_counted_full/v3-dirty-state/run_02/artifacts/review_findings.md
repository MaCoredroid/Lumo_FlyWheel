# Review Findings

## Switching the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:15-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged contract says nightly automation shells out without flags and expects JSON by default, but this change makes the no-flag CLI path print human-readable markdown instead. The PR description also says existing JSON behavior should stay intact, so downstream jobs that do not pass `--output json` will start receiving a different wire format.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, review/ci_snapshot.md

## The `json` renderer key now returns markdown instead of the JSON payload shape
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:7-7
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: The registry now routes `output="json"` to `render_markdown`, so `export_report(..., output="json")` returns a markdown string instead of the dictionary produced by the unchanged JSON renderer. In the CLI, that means `--output json` serializes a quoted markdown blob rather than the machine-readable object that the unchanged contract notes say callers depend on.
- evidence: review/patch.diff, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
