# Review Findings

## Changing the CLI default to markdown breaks the unflagged nightly consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. Flipping the parser default to markdown makes that unchanged caller receive prose instead of the historical object payload, which is the same failure mode called out in the rollback note.
- missing test: Visible CI only covered the markdown path while the downstream json integration suite remained skipped, so the historical no-flag contract was not exercised.
- evidence: review/pr_description.md, review/patch.diff, release_context/nightly_export_consumer.md, incident_context/inc_241_markdown_rollout_rollback.md, review/ci_snapshot.md

## The json renderer registry entry now returns markdown for explicit json requests
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/docs/export_contract.md
- impact: The PR description says existing JSON behavior should remain intact, but `_RENDERERS['json']` now points at `render_markdown`. Any unchanged caller that requests `output='json'` through `export_report` or the CLI now gets markdown text instead of the dictionary-like payload with a top-level `version` field that the contract documents require.
- missing test: There is no active test asserting explicit `--output json` behavior in this stack; the json contract notes exist, but the json contract suite was skipped.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
