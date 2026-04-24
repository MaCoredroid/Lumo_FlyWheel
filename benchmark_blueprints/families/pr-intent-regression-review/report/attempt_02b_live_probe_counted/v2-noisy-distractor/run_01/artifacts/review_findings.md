# Review Findings

## Changing the parser default to markdown breaks the no-flag JSON contract
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and depends on JSON output, so flipping the default to `markdown` changes the observable behavior of `release-readiness <input>` for existing callers instead of keeping markdown opt-in.
- evidence: review/patch.diff, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, review/ci_snapshot.md

## The json renderer registry entry now returns markdown for explicit json exports
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/renderers/json_renderer.py
- impact: The unchanged `render_json()` path still defines the machine-readable payload shape, but the registry now routes `output="json"` to `render_markdown()`, so explicit JSON exports return a string and the CLI serializes quoted markdown instead of the contract-preserving object.
- evidence: review/patch.diff, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/renderers/json_renderer.py, repo/tests/test_json_contract_notes.md
