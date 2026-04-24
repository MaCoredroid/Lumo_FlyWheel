# Review Findings

## Switching the CLI default to markdown breaks the flagless nightly consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout still invokes `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. Changing the default to markdown means that caller now receives a markdown document instead of machine-readable JSON, which is exactly the rollback scenario already documented in incident context.
- missing test: The CI snapshot explicitly skips `repo/tests/test_json_contract.py`, so the historical default-output contract is not exercised in this stack.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, incident_context/inc_241_markdown_rollout_rollback.md

## The renderer registry now returns markdown for explicit `--output json`
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The unchanged CLI still treats `args.output == "json"` as the machine-readable path and serializes whatever `export_report(..., output="json")` returns with `json.dumps`. After this registry change, explicit JSON requests get the markdown renderer instead, so the CLI emits a quoted markdown string and any direct `export_report(..., output="json")` caller receives a string rather than the documented dict-like payload.
- missing test: Only markdown snapshot coverage ran in CI; there is no active regression test in this stack for the explicit JSON renderer contract.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md
