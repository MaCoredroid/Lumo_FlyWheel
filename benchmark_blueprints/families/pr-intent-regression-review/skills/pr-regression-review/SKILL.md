# PR Regression Review

Use this skill when the task is a code review and the benchmark wants behavioral findings, not implementation work.

## Objective

Find regressions, invariant breaks, and missing tests that matter to the PR’s stated intent.

## Procedure

1. Read the PR description, then ignore its confidence.
2. Inspect changed lines first.
3. For each candidate issue, locate one unchanged dependent surface that the change can break.
4. Write only findings that have both a concrete location and a concrete impact.
5. Treat missing tests as findings only when a specific regression path is uncovered.

## Guardrails

- Style feedback is not a substitute for behavioral review.
- Passing visible CI is not proof that the PR matches its stated intent.
- Do not review untouched legacy code unless the changed surface directly violates it.

