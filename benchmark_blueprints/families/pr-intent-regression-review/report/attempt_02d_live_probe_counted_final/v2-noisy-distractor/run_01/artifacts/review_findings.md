# Review Findings

## Changing the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says downstream automation shells out without flags and expects JSON by default. After this change, `release-readiness <input>` emits the human markdown view instead, so existing no-flag jobs stop receiving a machine-readable object with a top-level `version` field.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, artifacts/runtime_checks.md

## The json registry entry now routes explicit JSON exports through the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/export.py
- impact: The unchanged `export_report(..., output="json")` API depends on the registry to preserve the historical dict payload. With `json` mapped to `render_markdown`, explicit JSON requests now return a markdown string, and the CLI wraps that string in JSON quotes instead of emitting the existing object shape that automation consumers parse.
- evidence: review/patch.diff, repo/src/release_readiness/export.py, repo/tests/test_json_contract_notes.md, artifacts/runtime_checks.md
