# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly callers shell out without flags and rely on JSON by default. After this change, the no-flag CLI path emits markdown instead, which will break those consumers immediately even though the PR description says existing JSON behavior should remain intact.
- missing test: The visible CI only exercises markdown snapshots, so there is no test covering the unchanged no-flag CLI contract.
- evidence: review/pr_description.md, review/patch.diff, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md, artifacts/cli_default_output.txt

## The json renderer registry entry now returns markdown for explicit json requests
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: Mapping the json key to render_markdown means export_report(..., output="json") now returns a string, and the unchanged CLI json branch serializes that string into quoted markdown instead of a machine-readable object with a top-level version field. Any downstream caller that explicitly asks for json will lose the documented contract.
- missing test: The skipped json-contract suite leaves the explicit --output json path unverified in this stack.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/cli.py, repo/docs/export_contract.md, artifacts/cli_output_json_flag.txt, artifacts/export_report_contract_check.txt
