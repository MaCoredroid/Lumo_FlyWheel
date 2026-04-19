# Evaluator Contract

- `family_id`: `pr-intent-regression-review`
- `task_id`: `t2_pr_review_markdown_registry_regressions`
- `total_points`: `100`
- `target_attack_band_for_naive_gpt_5_4_high`: `15-25`

## Scoring Breakdown

- `30` points: True behavioral regression findings that would materially block or reshape the PR.
- `20` points: True contract or invariant findings tied to the changed surface and an unchanged dependent surface.
- `15` points: Accurate line anchoring and impact explanation for each scored finding.
- `15` points: Evidence discipline and integrity under uncertainty, including refusal to fabricate unsupported findings.
- `10` points: Test-gap reasoning that is specific to the seeded behavioral gaps rather than generic “needs more tests.”
- `10` points: Review summary accurately judges whether the PR matches its stated intent.

## Caps

- Cap at `20/100` if the solver uses evidence from outside the provided PR bundle or repo snapshot.
- Cap at `10/100` if the review is dominated by style feedback or cosmetic nits.
- Cap at `20/100` if findings are plausible but not anchored to changed lines.
- Cap at `25/100` if findings mention changed lines but never tie them to an unchanged invariant-bearing surface.
- Cap at `30/100` if the review infers safety from CI without examining the behavioral contract.
- Cap at `15/100` if the test-gap section is generic rather than tied to a concrete missing regression guard.

## Automatic Failure Conditions

- `0` if the review contains no blocking or significant findings.
- `0-5` if required artifacts are missing.
- `0-5` if the review relies on files or line numbers not present in the diff bundle.

## Evidence Rules

- Only the provided PR bundle and repo snapshot count as evidence unless the task explicitly names an external surface.
- Every scored finding needs both a location and an impact statement.
- At least two scored findings must bridge changed code and unchanged dependent behavior.
- Findings about tests count only when tied to a specific missing behavioral check.

## Judging Procedure

1. Verify artifact presence and required count of findings.
2. Remove style-only comments from consideration.
3. Score substantive findings, then apply caps.
4. Judge the summary last.

## Meaningfulness Check

- A valid task instance needs at least two materially different review issues in different files or surfaces.
- If a solver can clear the task by reviewing the PR description alone, the authored instance is invalid.
