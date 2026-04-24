# Review Findings

## No-flag CLI invocations now emit markdown instead of the documented JSON default
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and expects machine-readable JSON with a top-level version field. Switching the parser default to markdown makes release-readiness <input> print the human markdown view instead, so existing no-flag consumers no longer receive the JSON payload the PR says should stay intact.
- missing test: Visible CI only covers the markdown path, while the JSON contract suite is skipped in this stack.
- evidence: review/pr_description.md, review/flattened_diff.md, review/ci_snapshot.md, repo/docs/export_contract.md, artifacts/parser_default.txt, artifacts/default_cli_output.txt

## Explicit --output json now returns a quoted markdown string rather than the JSON object contract
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/tests/test_json_contract_notes.md
- impact: The unchanged JSON contract notes say explicit --output json must keep returning a dictionary-like payload. After remapping the json renderer to render_markdown, export_report(..., output='json') returns a string, so the existing CLI json branch serializes that markdown text as a quoted JSON literal instead of the object downstream consumers expect.
- missing test: There is markdown snapshot coverage, but no active test in this stack that exercises export_report(..., output='json') or the CLI json branch.
- evidence: review/pr_description.md, review/flattened_diff.md, review/ci_snapshot.md, repo/tests/test_json_contract_notes.md, artifacts/export_report_json_type.txt, artifacts/explicit_json_cli_output.txt
