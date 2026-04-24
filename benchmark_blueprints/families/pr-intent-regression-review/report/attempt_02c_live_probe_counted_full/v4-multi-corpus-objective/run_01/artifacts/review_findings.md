# Review Findings

## Changing the default output to markdown breaks the nightly no-flag consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. After this change the default path emits markdown instead, so that downstream parse fails immediately.
- missing test: The CI snapshot says the downstream integration suite is skipped in this stack, so the no-flag automation contract is currently unguarded.
- evidence: review/pr_description.md, review/patch.diff, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, artifacts/default_cli_output.txt

## The json renderer registry now routes explicit --output json through markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/docs/export_contract.md
- impact: `export_report(..., output='json')` now returns a markdown string, and the CLI wraps that in `json.dumps`, producing a JSON string literal instead of the historical object payload. Callers that explicitly request `--output json` lose the documented machine-readable top-level fields such as `version`.
- missing test: The test changes add markdown snapshot coverage only, while the documented JSON contract remains untested in this stack.
- evidence: review/patch.diff, repo/src/release_readiness/cli.py, repo/tests/test_json_contract_notes.md, repo/docs/export_contract.md, artifacts/explicit_json_output.txt, artifacts/export_report_json_repr.txt
