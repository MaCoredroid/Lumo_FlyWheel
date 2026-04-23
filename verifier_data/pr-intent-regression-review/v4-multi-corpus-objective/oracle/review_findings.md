# Review Findings

## Changing the CLI default to markdown violates the stated JSON-compat contract
- severity: high
- changed: repo/src/release_readiness/cli.py:15-15
- linked surface: release_context/nightly_export_consumer.md
- impact: The PR description says existing JSON behavior stays intact, but the changed default flips no-flag invocations from machine-readable JSON to markdown output.
- missing test: There is no regression coverage for the default no-flag path even though automation relies on the historical JSON default.
- evidence: repo/src/release_readiness/cli.py, repo/docs/export_contract.md, review/pr_description.md, review/ci_snapshot.md, release_context/nightly_export_consumer.md

## The renderer registry now routes explicit json requests to markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:7-7
- linked surface: repo/src/release_readiness/export.py
- impact: Even callers that pass --output json will get a markdown string back because the registry binds the json key to render_markdown.
- missing test: The new tests only exercise markdown snapshots, so the explicit JSON path is no longer protected.
- evidence: repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/export.py, review/pr_description.md

## The added coverage only snapshots markdown and misses the two JSON regression paths
- severity: medium
- changed: repo/tests/test_markdown_export.py:4-4
- linked surface: repo/docs/export_contract.md
- impact: The test delta locks in the new markdown happy path but never checks either the no-flag default or explicit json behavior that the PR claims to preserve.
- missing test: Add one test for the default invocation and one for explicit --output json so the contract drift is caught.
- evidence: repo/tests/test_markdown_export.py, repo/docs/export_contract.md, review/ci_snapshot.md
