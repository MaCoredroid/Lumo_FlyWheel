# Plugin Contract Audit

Use this skill when a plugin loader is migrating to ESM and contract validation is part of the task.

## Workflow
1. Verify the real load path before touching docs.
2. Distinguish source-path fixes from built-output fixes.
3. Validate plugin shape at runtime rather than trusting import namespace shape.
4. Keep mixed-module helpers alive unless the task explicitly removes them.

## Anti-Patterns
- Source-only loader fixes.
- `default ?? mod` coercion with no runtime narrowing.
- Evidence or screenshot churn without actual command output.

## Done Signal
- Visible checks pass and the loader is contract-validating from built output.
