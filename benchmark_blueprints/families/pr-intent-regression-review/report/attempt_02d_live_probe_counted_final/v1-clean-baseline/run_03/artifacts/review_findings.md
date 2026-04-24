# Review Findings

## Zero-flag CLI invocations no longer return the documented JSON default
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: repo/docs/export_contract.md
- impact: Changing `build_parser()` to default `--output` to `markdown` breaks the unchanged CLI contract that nightly automation shells out without flags and expects a machine-readable object with a top-level `version` field. This is a behavioral regression against the PR's stated goal of preserving existing JSON behavior for automation consumers.
- missing test: Visible CI only exercised the markdown path while the JSON contract test remains skipped, so this default-format regression is currently unguarded.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, review/ci_snapshot.md

## Explicit `--output json` is rerouted to the markdown renderer
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The registry now maps `json` to `render_markdown`, but the unchanged `cli.main()` branch for `args.output == "json"` still serializes the renderer result with `json.dumps(...)`. As a result, explicit JSON requests emit a quoted markdown string instead of the dictionary produced by `render_json`, violating the existing automation-facing contract and the PR's promise that JSON behavior stays intact.
- missing test: The new coverage is markdown-only, and the skipped JSON contract suite means this explicit-format break can merge without a failing test.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/src/release_readiness/renderers/json_renderer.py, repo/docs/export_contract.md, review/ci_snapshot.md
