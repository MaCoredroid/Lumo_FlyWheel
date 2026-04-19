# Evaluator Contract: Nightly Regression Watch

## Purpose
Evaluate whether the solver repaired the nightly regression digest end to end rather than patching visible examples or wording only.

## Required submission surfaces
- Parser or digest code changes
- Automation config or prompt update for the existing nightly watch
- Runbook update
- Regenerated digest artifact

## Visible gates
1. Parser tests pass against the visible verifier payloads.
2. Digest rendering tests pass for one blocking night and one clean night.
3. Automation metadata tests confirm the existing nightly watch remains singular and active.

## Trusted checks
- Exactly one `nightly_regression_watch` automation definition remains active.
- Prompt and rendered output use `Action required` only for blocking conditions.
- Output artifact is regenerated from the current code path rather than accepted from a static fixture snapshot.

## Hidden checks
- `final_verdict.pass = true` plus missing required milestone must still page.
- Partial-success advisory runs must not page.
- Same-day reversal cases require “choose latest run” semantics, not “choose first blocker seen.”
- Mixed `milestones.results` object shapes must parse without a fixture-specific shim.
- Hidden fixture ordering and dates are permuted to reject hardcoded visible output.

## 100-point scoring breakdown
- `20` points: correct diagnosis of the schema rollover, blocker semantics, and duplicate-day risk.
- `25` points: observed parser and classification fix for `final_verdict.pass`, milestone results, and advisory versus blocking distinction. Proposed-only edits score at most `5/25`.
- `15` points: observed duplicate-day and latest-of-day selection behavior. Proposed-only edits score at most `3/15`.
- `10` points: observed automation repair without duplicating the watch. Proposed-only edits score at most `2/10`.
- `10` points: regenerated digest artifact that matches the repaired logic. No regenerated artifact yields `0/10`.
- `10` points: runbook alignment and wording contract. Proposed-only edits score at most `2/10`.
- `10` points: hidden robustness against shuffled inputs and mixed milestone shapes. No executed evidence yields `0/10`.

## Hard caps
- Cap at `20` if the solver creates a second automation instead of repairing the existing one.
- Cap at `20` if the digest artifact is hand-edited and not reproducible from shuffled hidden inputs.
- Cap at `30` if blocker classification ignores the required-milestone path.
- Cap at `25` for diagnosis-only or plan-only submissions that do not modify the concrete target workspace and regenerate the digest artifact.

## Shortcut defenses
- Compare emitted digest content with hidden shuffled inputs.
- Reject parser implementations that succeed only on flat visible key shapes.
- Reject output-only fixes that leave prompt or parser semantics stale.

## Final hardness judgment
- Current naive GPT-5.4/high outlook: under `30/100`.
- Confidence: medium.
- Main reason: the contract forces real classification plus dedupe and regeneration behavior.
