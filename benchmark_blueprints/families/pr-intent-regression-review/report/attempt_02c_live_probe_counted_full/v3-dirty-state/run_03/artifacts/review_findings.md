# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and depends on the default output being a machine-readable JSON object with a top-level `version` field. Flipping the parser default to `markdown` means those callers now receive human-readable markdown instead of the historical JSON payload, which is the opposite of the PR description's promise to keep existing JSON behavior intact for automation consumers.
- missing test: The only visible test coverage exercised here is markdown-only, and the downstream JSON-shellout suite is explicitly skipped.
- evidence: review/pr_description.md, review/patch.diff, review/ci_snapshot.md, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## The renderer registry now sends explicit json requests through the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged JSON contract notes say explicit `--output json` must continue returning a dictionary-like payload rather than a rendered string. After this mapping change, `export_report(..., output="json")` returns markdown text, so the CLI's unchanged `json.dumps(rendered, sort_keys=True)` path serializes that markdown as a quoted JSON string instead of an object. That removes the top-level fields automation consumers expect even when they explicitly pass `--output json`.
- missing test: No exercised test in this stack covers `output="json"`, and the integration suite that would catch the regression is skipped.
- evidence: review/pr_description.md, review/patch.diff, review/ci_snapshot.md, repo/tests/test_json_contract_notes.md, repo/src/release_readiness/renderers/json_renderer.py
