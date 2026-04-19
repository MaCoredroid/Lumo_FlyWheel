# Codex Route Contract

Use this skill when solving the `codex-config-skill-remediation` family.

## Workflow
1. Read `review/pr_077_threads.json` before trusting `review_summary.md`.
2. Extract the final reviewed route and tool-surface policy for `release-brief`.
3. Limit config edits to the workflow-local scope.
4. Align `SKILL.md` examples and rollout docs with the reviewed route.
5. Preserve validator integrity and existing global restrictions.

## Avoid
- widening the global allowlist
- weakening the validator
- browser-first examples after the route has been reviewed
- docs-only closure
