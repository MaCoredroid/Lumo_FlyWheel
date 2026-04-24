# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says downstream jobs shell out without flags and rely on JSON by default. After this change, the CLI prints markdown for `release-readiness <input>`, so those callers stop receiving a machine-readable payload with top-level fields like `version`.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, artifacts/runtime_checks.md

## The json renderer now returns markdown, so `--output json` loses the documented payload shape
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged contract notes require explicit `--output json` to stay dictionary-like. With `json` now mapped to `render_markdown`, `export_report(..., output='json')` returns a string, and the CLI wraps that string in `json.dumps(...)`, yielding a JSON string literal instead of the expected object payload.
- evidence: review/pr_description.md, review/patch.diff, repo/tests/test_json_contract_notes.md, artifacts/runtime_checks.md
