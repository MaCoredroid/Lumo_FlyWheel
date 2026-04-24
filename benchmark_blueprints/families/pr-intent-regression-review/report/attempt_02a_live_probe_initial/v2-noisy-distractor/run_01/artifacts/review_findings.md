# Review Findings

## Default CLI output no longer preserves the documented JSON contract
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: repo/docs/export_contract.md
- impact: Changing `--output` to default to `markdown` means callers that shell out without flags stop receiving the machine-readable object with a top-level `version` field that the unchanged export contract still promises for nightly automation consumers.
- missing test: The only exercised tests are markdown-only, and the skipped JSON contract suite leaves the default-output regression uncovered.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/docs/export_contract.md, review/ci_snapshot.md

## Explicit `--output json` now returns markdown text wrapped as JSON
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: The `json` registry entry now points at `render_markdown`, so `export_report(..., output="json")` returns a markdown string instead of the dictionary payload that explicit JSON consumers rely on. In the CLI this gets `json.dumps`-encoded into a quoted string, which drops the top-level `version` object shape entirely.
- missing test: There is no active test in this stack that exercises `output="json"`, so the renderer swap can pass the visible suite unnoticed.
- evidence: review/patch.diff, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
