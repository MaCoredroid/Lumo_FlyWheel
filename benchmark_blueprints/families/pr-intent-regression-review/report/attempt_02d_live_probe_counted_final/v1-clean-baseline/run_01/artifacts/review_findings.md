# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says callers shell out without flags and expect a machine-readable object with a top-level version field. After this change, invoking the CLI with no --output flag emits markdown instead, so existing automation consumers will stop receiving parseable JSON.
- missing test: Visible CI only covers markdown snapshots, and the JSON contract suite is explicitly skipped in this stack.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/docs/export_contract.md, artifacts/runtime_checks.md

## The json renderer key now dispatches to markdown output
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: export_report still defaults output='json' and the unchanged contract notes say explicit --output json must remain dictionary-like. With the registry now mapping json to render_markdown, the explicit json path returns a string, and the CLI serializes that as a quoted markdown blob instead of the prior JSON object shape.
- missing test: There is no active test in this stack asserting that output='json' returns the historical object payload.
- evidence: review/pr_description.md, review/patch.diff, repo/tests/test_json_contract_notes.md, artifacts/runtime_checks.md
