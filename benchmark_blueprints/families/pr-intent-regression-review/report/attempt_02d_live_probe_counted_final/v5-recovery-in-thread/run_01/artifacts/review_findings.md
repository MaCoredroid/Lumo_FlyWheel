# Review Findings

## Changing the CLI default to markdown breaks the existing no-flag automation path
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout still invokes `release-readiness input.json` without `--output`, so flipping the parser default to markdown changes stdout from the historical JSON object to human-readable markdown. That breaks the unchanged downstream parser that keys off the top-level `version` field and recreates the rollback scenario documented in the incident note.
- missing test: The visible CI only ran markdown coverage and explicitly skipped the integration/json contract suite, so the no-flag CLI contract is not exercised in this stack.
- evidence: review/pr_description.md, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, release_context/nightly_export_consumer.md, incident_context/inc_241_markdown_rollout_rollback.md, review/ci_snapshot.md

## The json registry entry now renders markdown instead of the JSON payload
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: Requests for `output="json"` now dispatch to `render_markdown`. The unchanged CLI json branch will therefore `json.dumps` a markdown string instead of the documented object payload, and any library caller using `export_report(..., output="json")` loses the dictionary-shaped contract that automation consumers still expect.
- missing test: This stack has no active assertion over the explicit JSON path: the JSON contract test is skipped in CI and the updated tests only cover markdown snapshots.
- evidence: review/pr_description.md, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
