# Review Findings

## Switching the default CLI output to markdown breaks the no-flag nightly consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout shells out to `release-readiness input.json` without `--output` and parses stdout as a JSON object with a top-level `version` field. After this change, the default stdout becomes markdown, so that downstream parse path fails even though the PR description says existing JSON behavior should stay intact.
- missing test: Visible CI only exercised the markdown snapshot path while the integration suite covering downstream consumers stayed skipped.
- evidence: review/pr_description.md, review/patch.diff, release_context/nightly_export_consumer.md, repo/docs/export_contract.md, review/ci_snapshot.md

## The renderer registry maps explicit json requests to the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/docs/export_contract.md
- impact: Callers that pass `--output json` no longer get the dictionary payload promised by the export contract. `export_report(..., output="json")` now returns markdown text, and `cli.main()` wraps that string in `json.dumps`, producing a quoted string instead of an object with `version`/`ready`/`services`, which breaks explicit JSON consumers as well as the stated compatibility goal.
- missing test: The added tests only cover `output="markdown"`; there is no active assertion that explicit JSON output still returns the historical object shape because the JSON contract suite is skipped in this stack.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md
