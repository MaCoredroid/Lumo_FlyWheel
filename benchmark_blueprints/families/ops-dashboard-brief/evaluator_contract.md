# Evaluator Contract: Ops Dashboard Brief

## Purpose
Evaluate whether the solver repaired the actual dashboard capture flow rather than relabeling text or reusing stale screenshots.

## Required submission surfaces
- Automation config or capture mapping
- Fresh screenshot artifacts and manifest
- Markdown brief
- Operator note

## Visible gates
1. Selector or capture-mapping tests pass.
2. Brief heading tests pass against the relabeled UI.
3. Screenshot manifest integrity tests pass.

## Trusted checks
- Capture manifest must contain a fresh run nonce or timestamp newer than the repaired automation config.
- Stale screenshots alone cannot satisfy the evidence contract.
- Operator note must mention the below-the-fold capture step.

## Hidden checks
- Irrelevant alert rail appears first; the true overnight-regressions panel must still be captured.
- Narrower viewport moves the target panel below the fold.
- Panel-content validation compares generated captures against grader-owned oracle evidence rather than filenames only.
- Brief headings must match the relabeled UI, not legacy section names.

## 100-point scoring breakdown
- `15` points: correct diagnosis of UI relabel, alert-rail confusion, and viewport-specific failure.
- `20` points: repaired automation config or capture mapping in the concrete workspace. Proposed-only edits score at most `4/20`.
- `20` points: fresh screenshot evidence and manifest proving regeneration. No fresh artifact yields `0/20`.
- `15` points: correct panel-content selection across wide and narrow viewport behavior. No browser-backed evidence yields `0/15`.
- `10` points: Markdown brief aligned to the relabeled UI. Proposed-only edits score at most `2/10`.
- `10` points: operator note aligned to the repaired capture path. Proposed-only edits score at most `2/10`.
- `10` points: hidden robustness against stale-artifact reuse and filename-only matching. No executed evidence yields `0/10`.

## Hard caps
- Cap at `20` if stale screenshots are reused as the only evidence.
- Cap at `20` if the operator note still omits the below-the-fold step.
- Cap at `30` if panel validation succeeds by filename only rather than content.
- Cap at `25` for diagnosis-only or plan-only submissions that do not produce fresh browser-backed artifacts from the concrete dashboard workspace.

## Shortcut defenses
- Require fresh artifacts, not just artifact references.
- Validate panel content and viewport behavior.
- Tie the brief and operator note to the actual capture flow.

## Final hardness judgment
- Current naive GPT-5.4/high outlook: under `30/100` if browser validation is implemented as written.
- Confidence: medium.
- Main reason: fresh-capture and content-validation requirements block browser-free rewrites.
