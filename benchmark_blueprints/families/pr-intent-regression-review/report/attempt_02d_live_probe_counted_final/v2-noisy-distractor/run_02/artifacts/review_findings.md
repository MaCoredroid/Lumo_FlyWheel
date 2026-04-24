# Review Findings

## Changing the CLI default to markdown breaks callers that shell out without flags
- severity: high
- changed: repo/src/release_readiness/cli.py:12-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says the CLI must keep json as the default because nightly automation shells out without flags and expects a machine-readable object with a top-level version field. After this change, invoking the CLI without --output prints the markdown document instead, so the default command path no longer satisfies the documented automation contract or the PR's promise to keep existing JSON behavior intact.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/docs/export_contract.md, review/ci_snapshot.md

## The json renderer registry entry now returns markdown instead of the JSON payload
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged JSON contract notes say explicit --output json must keep returning a dictionary-like payload, but this registry now maps the json key to render_markdown. The CLI's json branch still runs json.dumps(rendered), so the observed output becomes a quoted markdown string rather than an object with top-level fields like version. That is a behavioral regression for any downstream consumer that explicitly requests json.
- evidence: review/pr_description.md, review/flattened_diff.md, repo/tests/test_json_contract_notes.md, repo/src/release_readiness/cli.py
