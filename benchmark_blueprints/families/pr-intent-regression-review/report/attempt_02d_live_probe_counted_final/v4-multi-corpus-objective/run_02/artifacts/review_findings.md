# Review Findings

## Changing the CLI default to markdown breaks the nightly no-flag consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout still shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. After this default flip, the unchanged consumer receives markdown instead of a machine-readable object, so the historical automation path described in the PR stops working.
- evidence: review/pr_description.md, review/flattened_diff.md, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, artifacts/repro_default_cli.txt

## The `json` renderer now routes to markdown, so explicit `--output json` loses the object contract
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: Mapping `json` to `render_markdown` makes `export_report(..., output='json')` return a string. The unchanged CLI then `json.dumps` that string and emits a quoted markdown blob, so callers no longer receive the top-level `version` field and other dictionary keys that explicit JSON consumers rely on.
- evidence: review/flattened_diff.md, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, artifacts/repro_explicit_json_cli.txt, artifacts/repro_export_report_json.txt
