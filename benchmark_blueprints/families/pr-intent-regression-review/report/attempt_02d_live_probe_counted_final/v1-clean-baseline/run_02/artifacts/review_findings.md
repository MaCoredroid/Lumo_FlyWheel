# Review Findings

## Changing the CLI default to markdown breaks no-flag automation consumers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and depends on JSON being the default output. After this change, invoking the CLI without --output emits the markdown view instead, so existing consumers lose the top-level JSON object the PR description said would remain intact.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, repo/docs/export_contract.md, repo/tests/test_json_contract_notes.md

## The json renderer registry entry now returns markdown for explicit --output json
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The unchanged CLI json branch still assumes export_report(..., output="json") returns a dictionary-like payload and passes it to json.dumps. With the registry remapped to render_markdown, explicit --output json now produces a quoted markdown string instead of the prior machine-readable object, which breaks the explicit JSON compatibility guarantee.
- evidence: review/pr_description.md, review/patch.diff, review/flattened_diff.md, repo/src/release_readiness/cli.py, repo/src/release_readiness/renderers/json_renderer.py, repo/docs/export_contract.md
