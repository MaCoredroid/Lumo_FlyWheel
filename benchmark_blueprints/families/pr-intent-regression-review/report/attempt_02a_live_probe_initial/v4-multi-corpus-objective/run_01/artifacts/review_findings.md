# Review Findings

## Switching the CLI default to markdown breaks the nightly no-flag consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The nightly fanout still shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. After this change the default path prints markdown instead, so that unchanged downstream consumer can no longer parse the command output.
- missing test: The visible CI only exercises the markdown path, so there is no coverage for the historical no-flag CLI contract.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, review/ci_snapshot.md, release_context/nightly_export_consumer.md, repo/docs/export_contract.md

## The renderer registry now routes explicit json exports through the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: Callers that explicitly request `--output json` or call `export_report(..., output="json")` historically receive a dictionary-like payload. This mapping now returns the markdown string renderer instead, so the CLI emits a JSON-encoded string rather than the documented object contract and automation consumers lose the top-level fields they key on.
- missing test: This stack skips the JSON contract suite, and the added tests only assert markdown snapshots.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, repo/src/release_readiness/export.py, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md, repo/docs/export_contract.md
