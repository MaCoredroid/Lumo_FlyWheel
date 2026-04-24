# Review Findings

## Switching the CLI default to markdown breaks the unchanged nightly no-flag consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:15-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout still shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. After this change, the default invocation returns markdown instead, so that consumer will fail as soon as it reads the new default output.
- missing test: Visible CI only covers the markdown path while the downstream integration suite that exercises the no-flag consumer is skipped in this stack.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, artifacts/runtime_repro.md

## The renderer registry now routes `json` requests to the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:7-7
- linked surface: repo/tests/test_json_contract_notes.md
- impact: Calling `export_report(..., output="json")` no longer returns the dictionary-like payload described by the existing contract. The CLI now wraps the markdown string in `json.dumps(...)`, so explicit `--output json` callers receive a quoted markdown blob rather than the historical object with `version`, `ready`, and `services` fields.
- missing test: The updated test file only snapshots markdown output, so the explicit JSON contract documented in the unchanged notes is no longer exercised here.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, artifacts/runtime_repro.md
