# Review Findings

## Changing the CLI default to markdown re-breaks the flagless nightly consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout still shells out to `release-readiness input.json` with no `--output` flag and parses stdout as JSON keyed by the top-level `version` field. With this default flipped to markdown, the CLI now emits the human-readable markdown report on the flagless path, which is the same failure mode that already required the INC-241 rollback.
- missing test: There is still no non-skipped integration test that shells out without `--output` and asserts the historical JSON default.
- evidence: review/pr_description.md, repo/src/release_readiness/cli.py, release_context/nightly_export_consumer.md, incident_context/inc_241_markdown_rollout_rollback.md, repo/docs/export_contract.md

## The renderer registry makes explicit `--output json` return markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/docs/export_contract.md
- impact: The unchanged contract requires explicit `--output json` to remain byte-for-byte compatible and dictionary-like, but the registry now maps `json` to `render_markdown`. `export_report(..., output='json')` therefore returns a string, and the CLI's existing `json.dumps(rendered, sort_keys=True)` branch serializes that string into a quoted JSON string instead of an object with a top-level `version` field.
- missing test: The suite adds markdown snapshots, but it still lacks a regression test that exercises `output='json'` and asserts the returned object shape.
- evidence: review/pr_description.md, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md
