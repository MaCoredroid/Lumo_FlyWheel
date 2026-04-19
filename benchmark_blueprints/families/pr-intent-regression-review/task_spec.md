# PR Intent Regression Review

- `task_id`: `t2_pr_review_markdown_registry_regressions`
- `family_id`: `pr-intent-regression-review`
- `scenario_type`: `contextual_code_review`

## Task Prompt

Review a feature PR that claims to add Markdown export support for a release-readiness CLI without changing existing JSON behavior. You are given the repo, the PR diff, the author’s summary, and a CI snapshot. Do not fix the code. Leave high-signal review findings only.

Produce:

- `artifacts/review_findings.md`: 2-4 findings ordered by severity, each tied to exact files and lines in the diff.
- `artifacts/review_summary.md`: a short summary of whether the PR matches its stated intent and what tests or follow-ups are missing.

The bar is behavioral review. Style notes, naming preferences, and broad refactor suggestions should not appear unless they directly affect correctness.

## Workspace Bundle

- Medium Python repo with `src/`, `tests/`, `docs/`, and plugin-style renderer registration.
- PR bundle containing `pr_description.md`, `patch.diff`, and a flattened comment-free diff view.
- CI snapshot showing one passing visible suite and one skipped integration suite.
- Generated fixture churn and one docs update mixed into the same PR.

## Seeded Ambiguity

Authoring note: these are construction notes for the benchmark pack. The solver-visible PR bundle should contain the evidence, but not these explicit answer-shape hints.

- The PR narrative overstates how isolated the change is.
- One changed path preserves the happy path but breaks a shared or cross-surface invariant.
- Diff noise is large enough to reward superficial review behavior unless the grader is strict.
- Test coverage is part of the intended review signal, but only if it is tied to a concrete behavioral gap.

## Required Surfaces

- Diff inspection and review-comment style output.
- Repo search and base-code reading for invariant checking.
- Optional shell for targeted test or grep inspection.
- Evidence must come from the provided PR bundle and repo snapshot only. No sibling repos or unrelated workspace files may be used to fabricate line-anchored findings.
- No code editing.

## Expected Deliverables

- Findings with concrete file and line references.
- Each finding must explain impact, not just describe a diff.
- A short review summary that states whether the PR matches its stated intent.

## Grader Contract

- Full credit requires:
- At least one true behavioral regression finding.
- At least one true contract or invariance finding tied to the registry or CLI integration path.
- At least one missing-test observation if the seeded gap exists.
- Findings must target the actual changed lines or the immediate invariant they violate.
- At least two findings must cite both a changed line and one unchanged dependent surface whose invariant is affected.
- The review summary must correctly judge the PR’s stated intent against real behavior.
- Partial credit if one major regression is found but the review misses the second seeded issue or over-focuses on style.
- Low credit if the review is generic, unanchored, or dominated by nits.

## Red-Team Traps

- Spend the review budget on formatting churn or generated fixtures.
- Infer correctness from passing visible CI.
- Praise the happy path and miss the unchanged-JSON claim regression.
- File comments against untouched legacy code instead of the PR’s changed surface.
- Suggest a rewrite instead of articulating a concrete blocking finding.

## Quality Gate

- Hardening applied after adversarial probe:
- The runtime prompt and solver-visible setup should not list the seeded regression categories.
- The grader should reject review bundles that contain only style feedback.
- Hidden checks should require one finding about stated-intent drift and one about cross-surface contract drift.
- The seeded regressions should live in different files so “one lucky grep hit” does not clear the task.
- At least two scored findings must connect a changed line to an unchanged invariant-bearing surface, which makes diff-header bluffing much harder.
- The bundle boundary is now explicit, and honest refusal to fabricate unsupported line findings is scoreable but capped.
- Hardening target: a naive GPT-5.4/high solver should not clear 30/100 by posting plausible but unverified review nits.
- Probe record:
- Initial adversarial probe judged the spec too easy at roughly 35-55/100 because the public spec named the review buckets too directly.
- Hardening moved those buckets into hidden grading and raised the evidence bar for findings.
- A later live GPT-5.4/high family-bundle run refused to invent missing diff evidence; the evaluator now reserves limited points for evidence discipline rather than forcing fabricated findings.
- Current expectation after hardening: under 30/100 for a naive GPT-5.4/high solver if the grader enforces changed-line plus unchanged-surface linkage.
