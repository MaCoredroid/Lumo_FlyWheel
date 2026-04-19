# Evaluator Contract: Review Thread UI Hardening

## Purpose
Score whether a solver correctly triaged live review threads, repaired the true UI and accessibility defects, updated the right visual-test coverage, and mapped replies to the actionable review items.

## 100-Point Breakdown
- `20` Correct triage of unresolved versus resolved review threads.
- `20` Runtime UI repair of the real reviewed issue.
- `15` Accessibility correctness on the actual interactive control.
- `15` Correct viewport or config coverage for the reopened regression.
- `10` Review-reply mapping quality.
- `10` Fresh evidence from the repaired route and viewport.
- `10` Branch hygiene and avoidance of overbroad or unrelated edits.

## Scoring Caps
- Cap at `20/100` if the submission does not include both a runtime fix and the required config or coverage fix.
- Cap at `10/100` if the submission is reply-only, screenshot-only, or baseline-only.
- Cap at `25/100` if the solver edits the wrong thread set or cannot prove thread-to-fix mapping.
- Cap at `30/100` if the solver applies blanket accessibility-label changes instead of identifying the actual reviewed control.

## Visible Pass Signals
- Existing branch tests pass.
- One reopened mobile viewport case passes.
- Submission includes concise review replies and fresh evidence.

## Hidden Evaluation Signals
- Hidden checks validate unresolved-thread targeting and reject churn on resolved threads.
- Hidden checks ensure the accessible name lands on the actual interactive control.
- Hidden checks require the exact reopened viewport from the review export, not a plausible nearby one.
- Hidden checks reject clipping, truncation, or baseline-only fixes.
- Hidden checks punish overbroad accessibility or config edits outside the reviewed surface.

## Shortcut Failure Modes
- Reply to every thread without fixing code.
- Add the wrong viewport to test config.
- Blanket `aria-label` changes across all icon buttons.
- Infer the wrong route from filenames or stale screenshot names.
- Patch evidence artifacts only.

## Calibration Target
- Design-time family-package run target: `15-25/100`.
- Family freeze target for a naive GPT-5.4/high solver with the full benchmark bundle: roughly `20/100`, and not above `25/100`.

