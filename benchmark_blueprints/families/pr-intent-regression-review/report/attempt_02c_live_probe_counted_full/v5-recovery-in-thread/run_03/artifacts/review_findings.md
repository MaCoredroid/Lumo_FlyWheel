# Review Findings

## Changing the CLI default to markdown breaks the no-flag automation path
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout still invokes `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. With this default flipped to markdown, that unchanged consumer now receives a human-readable document instead of machine-readable JSON, which reproduces the rollback scenario documented in INC-241.
- evidence: review/pr_description.md, review/patch.diff, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, incident_context/inc_241_markdown_rollout_rollback.md, artifacts/repro_cli_output.txt

## The json renderer registry entry now returns markdown for explicit json exports
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/export.py
- impact: Unchanged callers still route `output='json'` through `export_report()`, expecting the dictionary shape produced by `render_json`. After this registry swap they get the markdown string instead, and the CLI's unchanged `json.dumps(rendered)` path serializes that string as quoted markdown rather than the historical object with `version`, `ready`, and `services` fields.
- evidence: review/patch.diff, repo/src/release_readiness/export.py, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, artifacts/repro_cli_output.txt
