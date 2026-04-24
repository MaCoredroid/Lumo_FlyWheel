# Review Findings

## Changing the CLI default to markdown breaks existing no-flag automation
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by a top-level `version` field. After this default flip, the command now emits markdown instead, so that consumer will fail on its normal path.
- missing test: Visible CI only exercised the markdown tests while the JSON integration suite stayed skipped, so this contract break is not covered in the stack.
- evidence: review/pr_description.md, repo/src/release_readiness/cli.py, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, review/ci_snapshot.md

## The json renderer registry entry now routes explicit JSON requests to markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/docs/export_contract.md
- impact: The unchanged contract says explicit `--output json` must remain machine-readable. With `json` mapped to `render_markdown`, `export_report(..., output="json")` returns a markdown string, and `cli.py` then `json.dumps` that string into a quoted blob instead of the historical object payload. Any caller expecting fields like `version` from explicit JSON output will break.
- missing test: The stack added markdown snapshots but did not keep a live JSON-path assertion, and the JSON integration suite is skipped here.
- evidence: repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/cli.py, repo/src/release_readiness/renderers/json_renderer.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
