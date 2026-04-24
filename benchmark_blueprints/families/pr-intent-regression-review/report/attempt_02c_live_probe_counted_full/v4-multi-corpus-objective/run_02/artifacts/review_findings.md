# Review Findings

## Changing the CLI default output to markdown breaks the flagless nightly consumer
- severity: high
- changed: repo/src/release_readiness/cli.py:15-16
- linked surface: release_context/nightly_export_consumer.md
- impact: The unchanged nightly fanout still shells out to `release-readiness input.json` without `--output` and parses stdout as JSON keyed by the top-level `version` field. Switching the parser default from `json` to `markdown` changes that flagless path into human-readable markdown, so the downstream consumer can no longer decode the command's default output.
- evidence: review/pr_description.md, review/patch.diff, release_context/nightly_export_consumer.md, repo/docs/export_contract.md

## The json renderer registry entry now returns markdown for explicit `--output json` callers
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged contract notes say explicit `--output json` must keep returning a dictionary-like payload, but this mapping now routes `json` to `render_markdown`. That means `export_report(..., output="json")` returns a markdown string instead of the structured object that automation consumers and the CLI's JSON path expect.
- evidence: review/patch.diff, repo/tests/test_json_contract_notes.md, repo/docs/export_contract.md, repo/src/release_readiness/renderers/json_renderer.py
