# Review Findings

## Switching the CLI default to markdown breaks the no-flag JSON contract
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: Unflagged `release-readiness input.json` now emits markdown instead of the historical JSON payload, which breaks the unchanged no-flag contract documented for automation consumers and the nightly fanout job described in the release context.
- missing test: The skipped integration coverage in `review/ci_snapshot.md` leaves the no-flag automation path unverified.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/docs/export_contract.md, release_context/nightly_export_consumer.md, incident_context/inc_241_markdown_rollout_rollback.md

## The json renderer registry entry now returns markdown for explicit `--output json`
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: Routing the `json` key to `render_markdown` makes `export_report(..., output='json')` return a string instead of the dict produced by the unchanged JSON renderer, so explicit JSON callers lose the machine-readable object the contract notes still require.
- missing test: The JSON contract suite is skipped in this stack, so there is no active test covering the explicit `output='json'` path.
- evidence: review/flattened_diff.md, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
