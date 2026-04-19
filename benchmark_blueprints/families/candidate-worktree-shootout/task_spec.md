# Task Spec: `candidate-worktree-shootout`

## Task Prompt

Choose the stronger of two plausible fixes for a report-filter normalization bug by evaluating them in isolated worktrees. Candidate A keeps normalization local to the CLI entrypoint. Candidate B moves normalization into the shared service layer. You may generate or inspect both in separate worktrees, but the final branch should land only one approach plus a comparison note that explains why the rejected option is weaker.

The bug appears in one visible CLI path, but at least one hidden regression exercises non-CLI callers.

## Workspace Bundle

- Small Python service repo with:
  - `src/report_filters/cli.py`
  - `src/report_filters/service.py`
  - `src/report_filters/normalization.py`
  - `tests/test_cli.py`
  - `tests/test_service.py`
  - `docs/filter_contract.md`
  - `artifacts/candidates/strategy_a_notes.md`
  - `artifacts/candidates/strategy_b_notes.md`
- Visible checks:
  - CLI tests fail on separator-heavy labels
  - broader service tests are partially hidden in the benchmark harness

## Seeded Orchestration Problem

The parent agent should isolate candidate implementations so local experimentation does not pollute the main workspace. The seeded trap is that one candidate can make visible CLI tests pass quickly while still leaving hidden non-CLI regressions; the stronger candidate must be identified from repo evidence rather than prompt hints. A naive solver that never evaluates alternatives can accidentally ship the shallow fix.

## Required Surfaces

- `subagents`
- `worktree`
- `shell`
- `apply_patch`

Preferred:

- `review`
- `screenshots`

## Deliverables

- Final integrated patch for the chosen strategy
- `artifacts/comparison/candidate_a_eval.md` with worktree path, exact commands run, touched files, and observed pass or fail results
- `artifacts/comparison/candidate_b_eval.md` with worktree path, exact commands run, touched files, and observed pass or fail results
- Comparison note summarizing both candidates, evidence gathered, and final selection
- Verification note naming both worktree directories and the exact validation command run in each

## Grader Contract

The task scores well only if the submission:

- evaluates both candidate strategies with concrete per-candidate evidence captured from isolated worktrees; unsupported narrative reasoning does not count
- lands the fix that satisfies both CLI and non-CLI callers, as demonstrated by repo evidence and validation
- explains why the rejected option passes visible tests but is still wrong
- adds or preserves a service-level regression test that fails under CLI-local normalization and passes under the chosen fix
- preserves workspace cleanliness by avoiding cross-candidate contamination
- updates docs if the chosen strategy changes the normalization contract location

Low-score behavior includes implementing only the visible quick fix, inventing a comparison note with no evidence, or mixing both strategies into an incoherent patch.

## Red-Team Traps

- Make visible CLI tests pass by normalizing only in `cli.py`
- Skip isolated worktrees and overwrite the main branch during experimentation
- Claim both candidates were considered without any concrete evidence
- Update docs to match the shallow fix instead of fixing the shared contract
- Blend partial pieces of both approaches into a hard-to-review patch
- Omit isolated worktree proof or leave normalization materially duplicated in `cli.py`

## Quality Gate

- Probe target: naive GPT-5.4/high should not exceed 30/100 with a one-path visible-test fix.
- Hardening applied after adversarial probe:
  - removed prompt leakage that named the winning strategy too directly
  - made per-worktree evidence artifacts mandatory rather than optional narrative comparison
  - added a service-level regression-test requirement for the non-CLI caller path
  - made isolated-worktree proof and shared-layer ownership part of correctness, with score caps implied by missing evidence
- Probe record:
  - probe model: child GPT-5.4/high adversarial review
  - initial verdict: over 30 too easy because the first draft leaked the likely winner and let comparison be faked in prose
- Current difficulty judgment: under 30 likely for a naive GPT-5.4/high solver after hardening
