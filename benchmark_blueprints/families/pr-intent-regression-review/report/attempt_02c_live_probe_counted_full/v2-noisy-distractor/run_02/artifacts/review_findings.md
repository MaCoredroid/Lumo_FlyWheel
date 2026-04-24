# Review Findings

## Switching the CLI default to markdown breaks no-flag automation callers
- severity: high
- changed: repo/src/release_readiness/cli.py:13-16
- linked surface: repo/docs/export_contract.md
- impact: The unchanged export contract says nightly automation shells out without flags and relies on JSON with a top-level version field. Changing the parser default to markdown makes the default CLI invocation emit human-readable text instead, so existing no-flag consumers stop receiving machine-readable JSON.
- missing test: Visible CI only exercised the markdown path, and the updated markdown tests never assert that invoking the CLI without --output still produces JSON.
- evidence: review/pr_description.md, review/patch.diff, review/ci_snapshot.md, repo/docs/export_contract.md, repo/src/release_readiness/cli.py

## The json renderer registry entry now routes explicit JSON requests to markdown
- severity: high
- changed: repo/src/release_readiness/renderers/registry.py:6-8
- linked surface: repo/src/release_readiness/cli.py
- impact: The unchanged CLI json branch still treats the rendered value as the automation payload and serializes it with json.dumps. Repointing the json registry entry at render_markdown makes export_report(..., output='json') return a string, so explicit --output json now prints a JSON-escaped Markdown blob instead of the dictionary-like payload callers expect.
- missing test: The changed tests cover only markdown snapshots; there is still no exercised assertion that export_report(..., output='json') returns the legacy object shape.
- evidence: review/pr_description.md, review/patch.diff, repo/src/release_readiness/renderers/registry.py, repo/src/release_readiness/export.py, repo/src/release_readiness/cli.py, repo/tests/test_json_contract_notes.md
